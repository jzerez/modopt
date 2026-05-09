import numpy as np
import time
import warnings

from modopt import Optimizer
from modopt.line_search_algorithms import Minpack2LS
# from modopt.merit_functions import AugmentedLagrangian
from modopt.core.merit_functions.uc_merit_function import UCMerit
from modopt.approximate_hessians import BFGSScipy
from modopt import CSDLAlphaProblem


class HVPUC(Optimizer):
    """
    A reconfigurable modified Newton optimizer
    developed in modOpt for unconstrained nonlinear optimization.

    Parameters
    ----------
    problem : Problem or ProblemLite
        Object containing the problem to be solved.
    recording : bool, default=False
        If ``True``, record all outputs from the optimization.
        This needs to be enabled for hot-starting the same problem later,
        if the optimization is interrupted.
    hot_start_from : str, optional
        The record file from which to hot-start the optimization.
    hot_start_atol : float, default=0.
        The absolute tolerance check for the inputs 
        when reusing outputs from the hot-start record.
    hot_start_rtol : float, default=0.
        The relative tolerance check for the inputs 
        when reusing outputs from the hot-start record.
    visualize : list, default=[]
        The list of scalar variables to visualize during the optimization.
    keep_viz_open : bool, default=False
        If ``True``, keeps the visualization window open after the optimization is complete.
    turn_off_outputs : bool, default=False
        If ``True``, prevent modOpt from generating any output files.

    maxiter : int, default=1000
        Maximum number of major iterations.
    opt_tol : float, default=1e-7
        Optimality tolerance.
    ls_min_step : float, default=1e-12
        Minimum step size for the line search.
    ls_max_step : float, default=1.
        Maximum step size for the line search.
    ls_maxiter : int, default=10
        Maximum number of iterations for the line search.
    ls_eta_a : float, default=1e-4
        Armijo (sufficient decrease condition) parameter for the line search.
    ls_eta_w : float, default=0.9
        Wolfe (curvature condition) parameter for the line search.
    ls_alpha_tol : float, default=1e-14
        Relative tolerance for an acceptable step in the line search.

    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'major', 'obj', 'x', 'opt',
        'time', 'nfev', 'ngev', 'step', 'low_curvature'.
    """
    # qp_tol : float, default=1e-4
    #     Tolerance for the QP subproblem.
    # qp_maxiter : int, default=5000
    #     Maximum number of iterations for the QP subproblem.
    def initialize(self):
        self.solver_name = 'hvpuc2'

        self.nx = self.problem.nx

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.hvp  = self.problem._compute_objective_hvp
        self.active_callbacks = ['obj', 'grad', 'obj_hvp']
        if self.problem.constrained:
            raise NotImplementedError(
                "HVP-UC is currently implemented for unconstrained optimization only. \
                 Stay tuned for the constrained version in the future!"
                 )

        ##################################################################
        # TEMPORARY: for CSDLAlphaProblem
        if type(self.problem) is CSDLAlphaProblem:
            self.obj = lambda x: self.problem._compute_objective(x, check_failure=True)
            self.grad = lambda x: self.problem._compute_objective_gradient(x, check_failure=True)
        ##################################################################

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-7, types=float)
        self.options.declare('readable_outputs', types=list, default=[])
        # self.options.declare('qp_maxiter', default=5000, types=int)

        self.options.declare('ls_min_step', default=1e-12, types=float)
        self.options.declare('ls_max_step', default=1.0, types=float)
        self.options.declare('ls_maxiter', default=10, types=int)
        self.options.declare('ls_eta_a', default=1e-4, types=float)
        self.options.declare('ls_eta_w', default=0.95, types=float)
        self.options.declare('ls_alpha_tol', default=1e-14, types=float)

        self.options.declare('m', default=1, types=int)
        self.options.declare('use_secant', default=False, types=bool)
        self.options.declare('use_exact_hess', default=False, types=bool)
        self.options.declare('method', default='AMS3', types=str)

        self.available_outputs = {
            'major': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),

            # Note that the number of constraints will
            # be updated after constraints are setup
            'opt': float,
            'time': float,
            'nfev': int,
            'ngev': int,
            'step': float,
            'low_curvature': int,
        }

    def setup(self):
        self.BFGS_exception_strategy = 'damp_update'
        # self.setup_constraints()
        nx   = self.nx

        self.successive_undefined_iterations = 0
        
        self.QN = BFGSScipy(nx=nx,
                            exception_strategy=self.BFGS_exception_strategy,
                            init_scale=1.0)
            
        self.MF = UCMerit(nx=nx,
                          f=self.obj,
                          g=self.grad)

        self.LSS = Minpack2LS(f=self.MF.compute_function,
                              g=self.MF.compute_gradient,
                              min_step=self.options['ls_min_step'],
                              max_step=self.options['ls_max_step'],
                              maxiter=self.options['ls_maxiter'],
                              eta_a=self.options['ls_eta_a'],
                              eta_w=self.options['ls_eta_w'],
                              alpha_tol=self.options['ls_alpha_tol'],
                              )

    def l1_penalty_line_search(self, x_k, x_qp, p_k, f_k, g_k):
        nx   = self.nx

        new_f_evals = 0
        converged = False
        # Penalty merit function value at alpha = alpha
        mf0 = f_k * 1.0

        exred = g_k.T @ x_qp

        ls_iter = 0
        step = 1.0
        alpha = 1.0
        alpha_min = 1e-1
        d = p_k[:nx] * 1.0
        while True:
            ls_iter += 1
            exred = alpha*exred # expected reduction (<0) in the merit function
            # Note that d needs to recursively scaled down: d != alpha * x_qp 
            d  = alpha * d
            x = x_k + d
            self.MF.update_functions_in_cache(['f'], x)
            f = self.MF.cache['f'][1]
            new_f_evals += 1

            # Penalty merit function value at alpha = alpha
            mf = f*1.0
            # Actual reduction in the merit function
            acred = mf - mf0

            # Break out of line search if the change (-ve) in the merit function 
            # is at least one-tenth of the expected change exred (always negative)
            # i.e., the obtained change in the merit function must be more negative 
            # than one-tenth of the expected reduction
            if (acred<=exred/10.0 or ls_iter > 10):
                new_g_evals = 0
                alpha = step * 1.0
                if acred <= exred:
                    converged = True
                break

            # Otherwise,
            alpha = np.maximum(exred/(2*(exred-acred)), alpha_min)
            step *= alpha

        return new_f_evals, converged, step

    def opt_check(self, g):
        opt_tol = self.options['opt_tol']

        opt_tol_factor = 1.
        # nonneg_violation = 0.
        # compl_violation = 0.
        stat_violation = np.linalg.norm(g, np.inf)

        
        scaled_opt_tol = opt_tol * opt_tol_factor
        # opt_check1 = (nonneg_violation <= scaled_opt_tol)
        # opt_check2 = (compl_violation <= scaled_opt_tol)
        opt_check3 = (stat_violation <= scaled_opt_tol)

        # opt is always nonnegative
        print('opt_tol_factor:', opt_tol_factor)
        print('stat_violation:', stat_violation)
        opt = stat_violation / opt_tol_factor
        opt_satisfied = opt_check3

        return opt_satisfied, opt

    def get_results_dict(self, x_k, f_k, opt, nfev, ngev, niter, time, success, err_msg):
        results = {'x': x_k,
                   'objective': f_k,
                   'optimality': opt,
                   'nfev': nfev,
                   'ngev': ngev,
                   'niter': niter,
                   'time': time,
                   'success': success, 
                   'err_msg': err_msg,}
        return results

    def solve(self):
        try:
            import qpsolvers
        except ImportError:
            raise ImportError("qpsolvers cannot be imported for the QP solver.  Install it with 'pip install qpsolvers'.")
        
        try:
            from quadprog import solve_qp
        except ImportError:
            raise ImportError("quadprog cannot be imported for the QP solver.  Install it with 'pip install quadprog'.")
        
        try:
            import highspy
        except ImportError:
            raise ImportError("HiGHS cannot be imported for the QP solver.  Install it with 'pip install highspy'.")

        # Assign shorter names to variables and methods
        nx = self.nx

        x0 = self.problem.x0
        maxiter = self.options['maxiter']
        # qp_maxiter = self.options['qp_maxiter']

        LSS = self.LSS
        QN = self.QN
        MF = self.MF

        eps = 2.22e-16

        start_time = time.time()

        # Proximal point initialization for a feasible start wrt bounds
        x_k = np.clip(x0, self.problem.x_lower, self.problem.x_upper)
        
        self.MF.update_functions_in_cache(['f', 'g'], x_k)
        f_k = self.MF.cache['f'][1]
        g_k = self.MF.cache['g'][1]
        undefined_proximal_point = False

        if np.isnan(f_k) or np.isinf(f_k):
            warnings.warn('Objective value at the computed proximal initial point is NaN or Inf. Trying given initial point ...')
            undefined_proximal_point = True
        elif np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)):
            warnings.warn('Objective gradient at the computed proximal initial point contains NaN or Inf. Trying given initial point ...')
            undefined_proximal_point = True
        else:
            print('Proximal point initialization is well-defined and good to go.')
        
        if undefined_proximal_point:
            if np.all(x0 == x_k):
                err_msg = 'Initial point provided and proximal point computed were the same and is undefined. Exiting ...'
                print(err_msg)
                return self.get_results_dict(x_k, f_k, None, 1, 1, 0, time.time() - start_time, False, err_msg)
            
            x_k = x0 * 1.

            self.MF.update_functions_in_cache(['f', 'g'], x_k)
            f_k = self.MF.cache['f'][1]
            g_k = self.MF.cache['g'][1]

            if np.isnan(f_k) or np.isinf(f_k):
                err_msg = 'Objective value at given initial point and computed proximal point is NaN or Inf. Exiting ...'
                print(err_msg)
                return self.get_results_dict(x_k, f_k, None, 2, 2, 0, time.time() - start_time, False, err_msg)
            elif np.any(np.isnan(g_k)) or np.any(np.isinf(g_k)):
                err_msg = 'Gradient at given initial point and computed proximal point contains NaN or Inf. Exiting ...'
                print(err_msg)
                return self.get_results_dict(x_k, f_k, None, 2, 2, 0, time.time() - start_time, False, err_msg)

        nfev = 1
        ngev = 1

        p_k  = np.zeros((nx,))

        # Iteration counter
        itr = 0
        n_hvp = 0
        all_Xs = np.array(x0.reshape(-1, 1))
        approx_hess = [QN.B_k,]
        nfevs = [nfev,]
        ngevs = [ngev,]
        nhvps = [n_hvp,]
        niters = [itr, ]
        objs = [f_k,]

        opt_satisfied, opt = self.opt_check(g_k)
        tol_satisfied = opt_satisfied

        # Initializing declared outputs
        self.update_outputs(major=0,
                            x=x_k,
                            obj=f_k,
                            opt=opt,
                            time=time.time() - start_time,
                            nfev=nfev,
                            ngev=ngev,
                            step=0.,
                            low_curvature=0)

        while itr < maxiter:
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate

            # Solve a strictly convex quadratic program
            # Minimize     1/2 x^T G x - a^T x
            # Subject to   C.T x >= b

            # def solve_qp(double[:, :] G, double[:] a, double[:, :] C=None, double[:] b=None, int meq=0, factorized=False):
            # First meq constraints are treated as equality constraints

            try:
                x_qp, f_qp, xu_qp, iter_qp, lag_qp, iact_qp = solve_qp(QN.B_k, -g_k)
                p_k[:] = x_qp

            except Exception as e:
                print('QP solver failed at major iteration:', itr)
                print(e)

                if "matrix G is not positive definite" in str(e):
                    print('Matrix G is not positive definite. Resetting Hessian.')
                    # Reset Hessian
                    wTw = np.dot(w_k, w_k)
                    wTd = np.dot(w_k, d_k[:nx])

                    init_scale = wTw / (wTd+1e-16) if wTd > 0 else 1.
                    self.QN = QN = BFGSScipy(nx=nx,
                                            exception_strategy=self.BFGS_exception_strategy,
                                            init_scale=init_scale)
                                            # init_scale=np.linalg.norm(np.diag(QN.B_k)))
                    
                    continue # Skip the rest of the iteration and solve QP with new reset Hessian

            # Clip the step length such that the design variables remain within bounds
            p_k[:nx] = np.clip(p_k, self.problem.x_lower - x_k, self.problem.x_upper - x_k)

            print('Major iteration:', itr)
            print("=====================================")

            dir_deriv_al_0 = np.dot(g_k, p_k)

            # Compute the step length along the search direction via a line search
            p_k_temp = p_k * 1.
            x_k_temp = x_k * 1.

            if dir_deriv_al_0 > -2.22e-16 or np.linalg.norm(p_k) <= 2.22e-16:
                alpha = 1.0
                mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = f_k, g_k, dir_deriv_al_0, 1, 1, True

            else:
                alpha, mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = LSS.search(
                    x=x_k_temp, p=p_k_temp, f0=f_k, g0=g_k)

            undefined_direction = False
            if np.isnan(mf_new) or np.isinf(mf_new) or np.isnan(mfg_new).any() or np.isinf(mfg_new).any():
                undefined_direction = True
                print('Merit function or its gradient at the new point has NaN or inf.')
            else:
                nfev += new_f_evals
                ngev += new_g_evals

            alpha_golden = 0.061803398875
            if (not converged) and (not undefined_direction):
                # Use an inexact LS with l1-penalty and only function evaluations
                new_f_evals, converged, alpha = self.l1_penalty_line_search(x_k, x_qp, p_k, f_k, g_k)

                nfev += new_f_evals
                
            if not undefined_direction:
                print("###### SUCCESSFUL LINE SEARCH #######")
                d_k = alpha * p_k
                d_k_temp = d_k * 1.

            g_old = g_k * 1.

            # If the line search failed with nans at alpha=1 earlier, find a smaller alpha where the function is well-defined
            undefined_new_point = False
            if undefined_direction:

                x_k_new = x_k + p_k[:nx]

                self.MF.update_functions_in_cache(['f', 'g'], x_k_new)
                f_new = self.MF.cache['f'][1]
                g_new = self.MF.cache['g'][1]
                alpha = 1.0
                new_f_evals = 0
                new_g_evals = 0

                while np.isnan(f_new) or np.isinf(f_new):
                    print('Objective function is NaN or Inf. Stepping back x_k_new.')
                    alpha *= 0.1
                    d_k_temp = alpha * p_k
                    x_k_new = x_k + d_k_temp[:nx]
                    self.MF.update_functions_in_cache(['f'], x_k_new)
                    f_new = self.MF.cache['f'][1]
                    new_f_evals += 1
                    if alpha < 1e-12:
                        undefined_new_point = True
                        break
                
                if not undefined_new_point:
                    while np.any(np.isnan(g_new)) or np.any(np.isinf(g_new)):
                        print('Objective gradient contains NaN or Inf. Stepping back x_k_new.')
                        alpha *= 0.1
                        d_k_temp = alpha * p_k
                        x_k_new = x_k + d_k_temp[:nx]
                        self.MF.update_functions_in_cache(['g'], x_k_new)
                        g_new = self.MF.cache['g'][1]
                        new_g_evals += 1
                        if alpha < 1e-12:
                            undefined_new_point = True
                            break

                if self.problem.constrained:
                    nfev += int(new_f_evals/2)
                    ngev += int(new_g_evals/2)
                else:
                    nfev += new_f_evals
                    ngev += new_g_evals
            
            if undefined_new_point:
                self.successive_undefined_iterations += 1
                if self.successive_undefined_iterations == 1:
                    print('No points along the search direction is well-defined. Resetting Hessian.')
                    self.QN = QN = BFGSScipy(nx=nx,
                                             exception_strategy=self.BFGS_exception_strategy,
                                             init_scale=1.)
                    continue

                if self.successive_undefined_iterations == 2:
                    err_msg = 'Two successive iterations with unsuccessful search along predicted direction for well-defined points. Terminating ...'
                    print(err_msg)
                    return self.get_results_dict(x_k, f_k, opt, nfev, ngev, itr, time.time() - start_time, False, err_msg)
     
            elif undefined_direction:
                self.successive_undefined_iterations = 0

                print('alpha successful without nan:', alpha)

                # If the line search failed with nans at alpha=1 earlier, reperform line search with a smaller alpha found above
                alpha_new, mf_new, mfg_new, mf_slope_new, new_f_evals, new_g_evals, converged = LSS.search(
                    x=x_k_temp, p=alpha*p_k, f0=f_k, g0=g_k)
                
                nfev += new_f_evals
                ngev += new_g_evals
                
                # If the line search failed to find a point satisfying the Wolfe conditions, try backtracking line search
                if not converged: 
                    print('Inside SLSQP line search: Reperforming backtracking line search with a smaller alpha found above.')
                    
                    new_f_evals, converged, alpha_new = self.l1_penalty_line_search(x_k, x_qp, alpha*p_k, f_k, g_k)
                    nfev += new_f_evals

                alpha *= alpha_new
                d_k = alpha * p_k
                d_k_temp = d_k * 1.

            # hvp_old = self.hvp(x_k, d_k_temp[:nx])
            # ngev += 1
            x_k += d_k_temp
            g_old = g_k * 1.

            self.MF.update_functions_in_cache(['f', 'g'], x_k)
            f_k = self.MF.cache['f'][1]
            g_k = self.MF.cache['g'][1]
            
            # HVP-related update for the BFGS Hessian approximation
            #######################################################
            QN_d_k = d_k[:nx]

            use_hvp = True
            if np.mod(itr, 1) == 0:
                if self.options['m']:
                    m = min(nx, self.options['m'])
                else:
                    m = 1

                if self.options['method'] == 'BFGS':
                    # self.QN = QN = BFGSScipy(nx=nx,
                    #         exception_strategy=self.BFGS_exception_strategy,
                    #         init_scale=1.0)
                    use_hvp = False
            else:
                m = 1
                if self.options['use_secant']:
                    use_hvp = False

            # orthogonal krylov HVPs with filtering and normalization
            # Set of HVP directions (inputs)
            S = np.ones((nx, m))

            # Set of HVPs (outputs)
            Y = np.ones(S.shape)

            s = np.copy(d_k_temp[:nx])
            invalid_idx = np.zeros((m, ), dtype=bool)

            if use_hvp:
                for i in range(m):
                    # correct direction by subtracting out directions we've already explored 
                    for j in range(i):
                        s -= np.dot(s, S[:, j]) * S[:, j]

                    if np.linalg.norm(s) < 1e-16 and i > 0:
                        invalid_idx[i] = True
                    
                    s /= np.linalg.norm(s)
                    y = self.hvp(x_k, s)
                    n_hvp += 1
                    
                    neg_curvature = np.dot(s, y) < 1e-4

                    if neg_curvature and 'skip' in self.BFGS_exception_strategy:
                        invalid_idx[i] = True

                    Y[:, i] = y
                    S[:, i] = s
                    s = y
                w_k = Y[:, 0]
            else:
                w_k = g_old - g_k
            # Skip directions of negative (or low) curvature
            # if 'skip' in self.BFGS_exception_strategy:
            m -= np.sum(invalid_idx)
            Y = np.delete(Y, invalid_idx, axis=1)
            S = np.delete(S, invalid_idx, axis=1)

            all_Xs = np.hstack((all_Xs, x_k.reshape(-1,1))) if all_Xs.size else x_k.reshape(-1,1)

            if self.options['method'] == 'BFGS':
                QN.update(d_k_temp[:nx], g_k - g_old)
                # B_k = QN.B_k
            elif self.options['method'] == 'iBFGS':
                if use_hvp:
                    for i in range(m):
                        QN.update(S[:, i], Y[:, i])
                else:
                    QN.update(d_k_temp[:nx], g_k - g_old)
                # B_k = QN.B_k
            
            approx_hess.append(QN.B_k.copy())

            # w_k = self.hvp(x_k, QN_d_k)
            # ngev += 1

            # QN.update(QN_d_k, w_k)
            # QN.update(QN_d_k, g_k - g_old)
            
            # # Second inplace BFGS update
            # w1 = w_k * 1.
            # d1 = d_k[:nx] * 1.
            # w1_ = w1 / (np.linalg.norm(w1) + 1e-16)
            # d1_ = d1 / (np.linalg.norm(d1) + 1e-16)
            # d2 = w1 - d1_ * (d1_.T @ w1)

            # if np.linalg.norm(d2) > 1e-16:
            #     w2 = self.hvp(x_k, d2)
            #     self.QN.update(d2, w2)

            #######################################################            

            # # <<<<<<<<<<<<<<<<<<<
            # # ALGORITHM ENDS HERE

            nfevs.append(nfev)
            ngevs.append(ngev)
            nhvps.append(n_hvp)
            niters.append(itr)
            objs.append(f_k)


            opt_satisfied, opt = self.opt_check(g_k)
            tol_satisfied = opt_satisfied
            low_curvature = False

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(
                major=itr,
                x=x_k,
                obj=f_k,
                opt=opt,
                time=time.time() - start_time,
                nfev=nfev,
                ngev=ngev,
                step=alpha,
                # step=0.5**ls_count,
                low_curvature=low_curvature,
                )
            if tol_satisfied:
                print('Convergence achieved!')
                break

        self.total_time = time.time() - start_time

        self.results = {
            'x': x_k,
            'objective': f_k,
            'objectives': objs,
            'optimality': opt,
            'nfevs': nfevs,
            'ngevs': ngevs,
            'niters': niters,
            'nfev': nfev,
            'ngev': ngev,
            'niter': itr,
            'time': self.total_time,
            'success': tol_satisfied, 
            'approx_hess': approx_hess,
            'true_hess': [],
            'bk_obj': [],
            'U': [],
            'ams_success': [],
            'n_hvps': nhvps,
            'n_hvp': n_hvp,
            'err_msg': '',
            'x_history': all_Xs,
        }

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results