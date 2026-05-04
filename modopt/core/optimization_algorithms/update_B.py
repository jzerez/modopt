import numpy as np
from scipy.optimize import minimize
import scipy 
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

"""
Different Strategies for updating the hessian approximation using HVP information.
"""


# This Method is equivalent to Direction 3 from 2-19 meeting notes
# updated on 2026-04-17. Moving multi-secant constraint to the objective as a penalty
# previous formulation is impossible to satisfy in all cases (esp when true hessian is not PD). 
class AdaptiveMultiSecant1():
    

    # Here, we're framing the update as an optimization problem
    # Want to find matrices U and V with the "least-square magnitude"
    # That satisfy:
    #   Our secant conditions
    #   a positive definite hessian update 
    def __init__(self):
        self.save_last = 0
        self.p     = 1.0 # 1.0 or 2.0. Param for computing distance weighting
        self.gamma = 1e-2   # Fudge factor for tuning weights. Close to 1 means distance doesn't impact weighting. Close to zero means that normal weighting applies. This term ensures a minimum influence from far away points. 
        self.all_X = None
        self.all_S = None
        self.all_Y = None
        self.all_m = None   # Stored HVPs per step
        self.weights = None
        self.prev_U = None
        self.prev_V = None
        self.prev_B = None
        self.n_itr = 0
        self.x0 = None

        # jax_obj = lambda U, V: jnp.linalg.norm(U @ U.T - V @ V.T, ord='fro')

        def jax_obj(U, V, alpha, beta, B, S_all, Y_all, gamma, weights):
            # find lowest magnitude of U. 
            # pow = jnp.min(jnp.abs(U))
            obj = jnp.linalg.norm(U @ alpha @ U.T - V @ beta @ V.T, ord='fro')
            B_new = B + U @ alpha @ U.T - V @ beta @ V.T

            c_all = weights*(B_new @ S_all - Y_all)

            penalized_obj = gamma * obj + jnp.sum(jnp.linalg.norm(c_all, axis=0))
            return penalized_obj
        
        _obj = jax.jit(jax_obj)
        self._obj  = lambda U, V, alpha, beta, B, S_all, Y_all, gamma, weights: np.float64(_obj(U, V, alpha, beta, B, S_all, Y_all, gamma, weights))

        _grad = jax.jit(jax.grad(jax_obj, argnums=[0, 1]))
        self._grad  = lambda U, V, alpha, beta, B, S_all, Y_all, gamma, weights: np.asarray(_grad(U, V, alpha, beta, B, S_all, Y_all, gamma, weights)).ravel()

        # Multi-secant condition. Replace with penalty term
        # IDEA: if we filter out HVPs, then maybe we can re-introduce this constraint. Or enforce it implicitly
        # Use a similar filter to block BFGS based on pseudo-eigenvalue decomp. 
        # If we find directions with negative curvature, we can/should ignore those directions (what do we lose by doing htis?)
        # If we filter out the directions with negative curvature, then we should be able strictly enforce both positive definiteness
        # AND secant condition, but only for HVPs taken at the current step. Idt we can guarantee secant conformity with previously taken HVPs
        # 
        # def jax_con1(U, V, B, S, Y):
        #     B_new = B + U @ U.T - V @ V.T
        #     return jnp.ravel(B_new @ S - Y)
        # _con1 = jax.jit(jax_con1)
        # self._con1  = lambda U, V, B, S, Y: np.array(_con1(U, V, B, S, Y))

        # Forcing positive-definiteness
        def jax_con2(U, V, alpha, beta, B):
            B_new = B + U @ alpha @ U.T - V @ beta @ V.T
            w = jnp.real(jnp.linalg.eigvals(B_new))
            return jnp.array([jnp.min(w) - 1e-6])
        _con2 = jax.jit(jax_con2)
        self._con2  = lambda U, V, alpha, beta, B: np.array(_con2(U, V, alpha, beta, B))

        # _jac1 = jax.jit(jax.jacobian(jax_con1, argnums=[0,1]))
        # def temp_jac1(U, V, B, S, Y):
        #     dU, dV = _jac1(U, V, B, S, Y)
        #     return np.hstack((dU.reshape(dU.shape[0], -1), dV.reshape(dV.shape[0], -1)))
        # self._jac1 = temp_jac1

        _jac2 = jax.jit(jax.jacobian(jax_con2, argnums=[0, 1]))
        def temp_jac2(U, V, alpha, beta, B):
            dU, dV = _jac2(U, V, alpha, beta, B)
            return np.hstack((dU.reshape(dU.shape[0], -1), 
                              dV.reshape(dV.shape[0], -1),
                              ))
        self._jac2 = temp_jac2

    def update_B(self, B, S, Y, x_k, r=1):
        nx, m = S.shape

        # NOTE: should we handle r and m separately? Right now we treat them as equal. 
        # What is optimal relationship between the two? 
        # r_k+1 >= m_k+1 (Rank of update needs to at least be equal to number of HVPs)
        # r_k+1 < sum_i m_i (Rank of update needs to be less than the total number of HVPs. Why?)
        if self.all_S is None:
            self.all_S = S * 1.0
            self.all_Y = Y * 1.0
            self.all_X = x_k.reshape(-1,1) * 1.
            self.x0 = x_k.reshape(-1,1) * 1.
            self.weights = np.ones((m,), dtype=np.float64)
            self.all_m = np.array([m,])
            r = min(m, nx)
        else:
            # # This is requried for when the number of previous steps is less than self.save_last
            # save_last = min(np.int32(self.all_S.shape[1]/r), self.save_last)
            # Store history for S, Y, and X
            self.all_m = np.hstack((m, self.all_m[:self.save_last]))
            m_total = np.sum(self.all_m)

            self.all_S = np.hstack((S, self.all_S))

            ri = -1
            while self.all_S.shape[1] > m_total:
                self.all_S = np.delete(self.all_S, ri, axis=1)
                
            self.all_Y = np.hstack((Y, self.all_Y))
            while self.all_Y.shape[1] > m_total:
                self.all_Y = np.delete(self.all_Y, ri, axis=1)
            

            self.all_X = np.hstack((x_k.reshape(-1,1), self.all_X))
            while self.all_X.shape[1] > self.save_last + 1:
                self.all_X = np.delete(self.all_X, ri, axis=1)

            dX = x_k.reshape(-1,1) - self.all_X
            distance = np.linalg.norm(dX, axis=0)
            self.weights = np.repeat(1.0 / (1.0 + distance**self.p) / np.exp(distance), self.all_m)
            r = max(min(m_total, nx, 10), 1)

        r1 = r
        r2 = r

        self.weights = self.gamma + (1-self.gamma) * self.weights

        # gamma is how much we care about the norm of UU^T. 
        # Here we are saying that the update for B should be dominated by
        # satisfying a weighted sum of past secant conditions.
        if len(self.weights) == 0:
            gamma = 1
        else:
            gamma = np.min(self.weights) * 0.1
        
        # Seed optimizer with previous U matrix, if available. 
        x0 = np.ones(((r1 + r2) * nx,))
        x0[:r1*nx] = 2


        # U and V represent sets of dyads. U adds new curvature (from latest batch of y) and V removes curvature (from last batch)
        # U analagous to Y, V analagous to B_k S
        # if not (self.prev_U is None):
        #     nu = self.prev_U.shape[1]
        #     x0[0:min(r1, nu)*nx] = np.reshape(self.prev_U, (nu*nx))[:min(nu, r1) * nx]

        # if not (self.prev_V is None):
        #     nv = self.prev_V.shape[1]
        #     x0[r1*nx:r1*nx + min(r2, nv)*nx] = np.reshape(self.prev_V, (nv*nx))[:min(nv, r2) * nx]

        # init U0 to be Y, alpha0 to be (S.T @ Y)^-1
        x0[:r1*nx] = self.all_Y.flat
        x0[r1*nx:] = (B @ self.all_S).flat
        alpha = np.linalg.inv(self.all_S.T @ Y)
        beta = np.linalg.inv(self.all_S.T @ B @ self.all_S)

        # init V0 to be B_k @ S, beta0 t obe S.T @ 

        # 


        if self.prev_B is None:
            self.prev_B = np.eye(nx)

        obj  = lambda x: self._obj(x[:r1*nx].reshape(nx, r1), x[r1*nx:].reshape(nx, r2), alpha, beta, B, self.all_S, self.all_Y, gamma, self.weights)
        grad = lambda x: self._grad(x[:r1*nx].reshape(nx, r1), x[r1*nx:].reshape(nx, r2), alpha, beta, B, self.all_S, self.all_Y, gamma, self.weights)


        constraints = {
            # Enforce positive definiteness
            'type': 'ineq',
            'fun': lambda x: self._con2(x[:r1*nx].reshape(nx, r1), 
                                        x[r1*nx:].reshape(nx, r2), 
                                        alpha,
                                        beta,
                                        B),
            'jac': lambda x: self._jac2(x[:r1*nx].reshape(nx, r1), 
                                        x[r1*nx:].reshape(nx, r2), 
                                        alpha, 
                                        beta,
                                        B)
        }

        solver_options = {
            'maxiter': 1000,
            'ftol': 1e-5,
            'disp': False,
        }

        results = minimize(
            obj,
            x0,
            method='SLSQP',
            jac=grad,
            constraints=constraints,
            options=solver_options,
            )

        U = np.reshape(results.x[:nx*r1], (nx, r1))
        V = np.reshape(results.x[nx*r1:], (nx, r2))

        if np.linalg.matrix_rank(U) < U.shape[1]:
            print('U is singular!')
        else:
            self.prev_U = U

        if np.linalg.matrix_rank(V) < V.shape[1]:
            print('V is Singular!')
        else:
            self.prev_V = V

        if 'Positive directional' in results['message']:
            print('inner optimization cannot find descent direction')

        if results['success']:
            
            B_new = B + U @ alpha @ U.T - V @ beta @ V.T
            # try: 
            #     _ = np.linalg.cholesky(B_new)
            #     self.prev_U = U
            #     self.prev_B = B_new
            # except np.linalg.LinAlgError:
            #     print('B_new is not Positive Definite!')            
        else:
            B_new = B
            print(results['message'])

        print(f"Success of multi-secant update: {results['success']}")
        self.n_itr += 1
        return B_new, results, U

# This method is mostly equivalent to Direction 4 in the 2-19 Notes
# Not sure about the initialization of B: 0.5*(1e-6 * np.eye(nx)) + 0.5*B
class AdaptiveMultiSecant2():
    def __init__(self):
        # Here, we're framing the update as an optimization problem
        # Want to find matrix U with the "least-square magnitude"
        # That satisfy:
        #   Our secant conditions
        #   a positive definite hessian update 
        jax_obj = lambda U: jnp.linalg.norm(U @ U.T, ord='fro')
        _obj = jax.jit(jax_obj)
        self._obj  = lambda U: np.float64(_obj(U))

        _grad = jax.jit(jax.grad(jax_obj, argnums=[0]))
        self._grad  = lambda U: np.asarray(_grad(U)[0].ravel())
        
        def jax_con(U, B, S, Y):
            B_new = B + U @ U.T
            return jnp.ravel(B_new @ S - Y)
        _con = jax.jit(jax_con)
        self._con  = lambda U, B, S, Y: np.array(_con(U, B, S, Y))

        _jac = jax.jit(jax.jacobian(jax_con, argnums=[0]))
        def temp_jac(U, B, S, Y):
            dU = _jac(U, B, S, Y)[0]
            return np.asarray(dU.reshape(dU.shape[0], -1))
        self._jac = temp_jac

    def update_B(self, B, S, Y, r=1):
        nx = S.shape[0]
        x0 = np.ones((r*nx,))

        # Initialize B as some scaled identity matrix (for positive definitness), 
        # plus scaled-down version of the current hessian approx
        B = 0.5*(1e-6 * np.eye(nx)) + 0.5*B

        # Unpack values of U from x0
        obj  = lambda x: self._obj(x.reshape(nx, r))
        grad = lambda x: self._grad(x.reshape(nx, r))

        constraints = {
            'type': 'eq',
            'fun': lambda x: self._con(x.reshape(nx, r), B, S, Y),
            'jac': lambda x: self._jac(x.reshape(nx, r), B, S, Y),
        }

        solver_options = {
            'maxiter': 1000,
            'ftol': 1e-6,
            'disp': False,
        }

        results = minimize(
            obj,
            x0,
            method='SLSQP',
            jac=grad,
            constraints=constraints,
            options=solver_options,
            )

        U = np.reshape(results.x, (nx, r))

        if results['success']:
            B_new = B + U @ U.T
        else:
            B_new = B

        print(f"Success of multi-secant update: {results['success']}")

        return B_new, results['success']

# Equivalent to direction 2 on 03-04 Notes 
class AdaptiveMultiSecant3():
    def __init__(self):
        self.save_last = 25
        self.p     = 1.0 # 1.0 or 2.0. Param for computing distance weighting
        self.gamma = 1e-2   # Fudge factor for tuning weights. Close to 1 means distance doesn't impact weighting. Close to zero means that normal weighting applies. This term ensures a minimum influence from far away points. 
        self.all_X = None
        self.all_S = None
        self.all_Y = None
        self.all_m = None   # Stored HVPs per step
        self.weights = None
        self.prev_U = None
        self.prev_B = None
        self.n_itr = 0
        self.x0 = None

        # NOTE: Need to add something here to prevent/catch overflow values of U. 
        def jax_obj(U, B, S_all, Y_all, beta, weights):
            # find lowest magnitude of U. 
            # pow = jnp.min(jnp.abs(U))
            pow = 1.0
            obj = jnp.linalg.norm(U/pow @ U.T/pow, ord='fro') * pow**2
            B_new = B + U @ U.T
            c_all = weights*(B_new @ S_all - Y_all)

            penalized_obj = beta * obj + jnp.sum(jnp.linalg.norm(c_all, axis=0))
            # penalized_obj = beta * obj + jnp.linalg.norm(c_all, ord='fro')
            return penalized_obj

        _obj = jax.jit(jax_obj)
        self._obj  = lambda U, B, S_all, Y_all, beta, weights: np.float64(_obj(U, B, S_all, Y_all, beta, weights))

        _grad = jax.jit(jax.grad(jax_obj, argnums=[0]))
        self._grad  = lambda U, B, S_all, Y_all, beta, weights: np.asarray(_grad(U, B, S_all, Y_all, beta, weights)[0].ravel())

    def update_B(self, B, S, Y, x_k, r=1):
        nx, m = S.shape
        

        # NOTE: should we handle r and m separately? Right now we treat them as equal. 
        # What is optimal relationship between the two? 
        # r_k+1 >= m_k+1 (Rank of update needs to at least be equal to number of HVPs)
        # r_k+1 < sum_i m_i (Rank of update needs to be less than the total number of HVPs. Why?)
        if self.all_S is None:
            self.all_S = S * 1.0
            self.all_Y = Y * 1.0
            self.all_X = x_k.reshape(-1,1) * 1.
            self.x0 = x_k.reshape(-1,1) * 1.
            self.weights = np.ones((m,), dtype=np.float64)
            self.all_m = np.array([m,])
            r = min(m, nx)
        else:
            # # This is requried for when the number of previous steps is less than self.save_last
            # save_last = min(np.int32(self.all_S.shape[1]/r), self.save_last)
            # Store history for S, Y, and X
            self.all_m = np.hstack((m, self.all_m[:self.save_last]))
            m_total = np.sum(self.all_m)

            self.all_S = np.hstack((S, self.all_S))

            ri = -1
            while self.all_S.shape[1] > m_total:
                self.all_S = np.delete(self.all_S, ri, axis=1)
                
            self.all_Y = np.hstack((Y, self.all_Y))
            while self.all_Y.shape[1] > m_total:
                self.all_Y = np.delete(self.all_Y, ri, axis=1)
            

            self.all_X = np.hstack((x_k.reshape(-1,1), self.all_X))
            while self.all_X.shape[1] > self.save_last + 1:
                self.all_X = np.delete(self.all_X, ri, axis=1)

            dX = x_k.reshape(-1,1) - self.all_X
            distance = np.linalg.norm(dX, axis=0)
            self.weights = np.repeat(1.0 / (1.0 + distance**self.p) / np.exp(distance), self.all_m)
            r = max(min(m_total, nx, 10), 1)
            # r = 10
            # self.weights = np.repeat(1.0 / np.exp(distance*self.p), r)
        if self.n_itr > self.save_last:
            print(self.weights)
        # Problem with the weights. "save last" is ambiguous. It should refer to the number of 
        # previous HVPs. But this breaks down when the number of HVPs is not consistent between steps. 
        # We will run into problems when nx > 6. The first step will use at least 6 HVPs, rather than 3. 
        self.weights = self.gamma + (1-self.gamma) * self.weights

        # Beta is how much we care about the norm of UU^T. 
        # Here we are saying that the update for B should be dominated by
        # satisfying a weighted sum of past secant conditions.
        beta = np.min(self.weights) * 0.001
        
        # Seed optimizer with previous U matrix, if available. 
        x0 = np.ones((r*nx,))

        if not (self.prev_U is None):
            nu = self.prev_U.shape[1]
            x0[:nu*nx] = np.reshape(self.prev_U, (nu*nx))

        # x0 = np.ones((r*nx,))
        B = 1e-6 * np.eye(nx)

        obj  = lambda x: self._obj(x.reshape(nx, r), B, self.all_S, self.all_Y, beta, self.weights)
        grad = lambda x: self._grad(x.reshape(nx, r), B, self.all_S, self.all_Y, beta, self.weights)

        solver_options = {
            'maxiter': 1000,
            'ftol': 1e-5,
            'disp': False,
        }

        results = minimize(
            obj,
            x0,
            method='SLSQP',
            jac=grad,
            options=solver_options,
            )

        U = np.reshape(results.x, (nx, r))
        if np.all(np.isclose(U - U[:, 0].reshape(-1, 1), 0)):
            print('U is rank 1!')

        if results['success']:
            
            B_new = B + U @ U.T
            try: 
                _ = np.linalg.cholesky(B_new)
                self.prev_U = U
                self.prev_B = B_new
            except np.linalg.LinAlgError:
                print('B_new is not Positive Definite!')            
        else:
            B_new = B
            print(results['message'])

        print(f"Success of multi-secant update: {results['success']}")
        self.n_itr += 1
        return B_new, results, U
    



# Equivalent to direction 2 on 03-04 Notes 
class BlockBFGS():
    def __init__(self):
        self.save_last = 0
        self.p     = 1.0 # 1.0 or 2.0. Param for computing distance weighting
        self.gamma = 1e-2   # Fudge factor for tuning weights. Close to 1 means distance doesn't impact weighting. Close to zero means that normal weighting applies. This term ensures a minimum influence from far away points. 
        self.all_X = None
        self.all_S = None
        self.all_Y = None
        self.all_m = None   # Stored HVPs per step
        self.weights = None
        self.prev_U = None
        self.B = None
        self.n_itr = 0
        self.x0 = None
        self.tau = 1e-4
        self.hess = None
        

        # # NOTE: Need to add something here to prevent/catch overflow values of U. 
        # def jax_obj(U, B, S_all, Y_all, beta, weights):
        #     # find lowest magnitude of U. 
        #     # pow = jnp.min(jnp.abs(U))
        #     pow = 1.0
        #     obj = jnp.linalg.norm(U/pow @ U.T/pow, ord='fro') * pow**2
        #     B_new = B + U @ U.T
        #     c_all = weights*(B_new @ S_all - Y_all)

        #     penalized_obj = beta * obj + jnp.sum(jnp.linalg.norm(c_all, axis=0))
        #     # penalized_obj = beta * obj + jnp.linalg.norm(c_all, ord='fro')
        #     return penalized_obj

        # _obj = jax.jit(jax_obj)
        # self._obj  = lambda U, B, S_all, Y_all, beta, weights: np.float64(_obj(U, B, S_all, Y_all, beta, weights))

        # _grad = jax.jit(jax.grad(jax_obj, argnums=[0]))
        # self._grad  = lambda U, B, S_all, Y_all, beta, weights: np.asarray(_grad(U, B, S_all, Y_all, beta, weights)[0].ravel())

    def filter_steps(self, tau=0.1, cutoff=True):
        n, n_steps = self.all_S.shape

        proj = self.all_S.T @ self.all_Y
        lu, sigma, perm = scipy.linalg.ldl(proj)


        keep = np.zeros([n_steps,], dtype=bool)

        for i, s in enumerate(self.all_S.T):
            if sigma[i, i] >= tau * s.T @ s:
                keep[i] = True

        nd = np.sum(keep)

        if cutoff and nd > n:
            vals = np.diag(sigma)
            idx = np.argsort(vals)[:-n]

            kept_idx = np.where(keep)[0]
            keep[kept_idx[idx]] = False
        

        return keep
    
    def filter_steps2(self, tau=0.1, cutoff=True):
        n, n_steps = self.all_S.shape

        D = self.all_S.copy()
        L = np.zeros((n_steps, n_steps))
        Sigma = np.zeros((n_steps, n_steps))
        A = D.T @ self.hess @ D
        # A = D.T @ self.all_Y
        A = D.T @ self.all_Y
        # A = 0.5 * (D.T @ self.all_Y + self.all_Y.T @ D)

        keep = np.zeros([n_steps,], dtype=bool)
        di = 0
        for i, s in enumerate(self.all_S.T):

            

            sigma = A[i, i] - np.sum([L[di, j]**2 * Sigma[j, j] for j in range(di)])
            if sigma >= tau * s.T @ s:
                Sigma[di, di] = sigma
                L[di, di] = 1
                for j in range(i+1, n_steps):
                    L[j, di] = 1/sigma * (A[j, i] - np.sum([L[di, k] * L[j, k] * Sigma[k, k] for k in range(di)]))
                di += 1
                keep[i] = True

        nd = np.sum(keep)
        # Take the elements of keep that correspond to the n-largest Sigma values
        if cutoff and nd > n:
            vals = np.diag(Sigma)[:nd]
            idx = np.argsort(vals)[:-n]

            kept_idx = np.where(keep)[0]
            keep[kept_idx[idx]] = False
        
        return keep


    
    def update_B(self, B, S, Y, x_k, r=1, tau=0.1):
        nx, m = S.shape

        if self.hess is None:
            self.hess = np.eye(nx)
        

        # NOTE: should we handle r and m separately? Right now we treat them as equal. 
        # What is optimal relationship between the two? 
        # r_k+1 >= m_k+1 (Rank of update needs to at least be equal to number of HVPs)
        # r_k+1 < sum_i m_i (Rank of update needs to be less than the total number of HVPs. Why?)
        if self.all_S is None:
            self.all_S = S * 1.0
            self.all_Y = Y * 1.0
            self.all_X = x_k.reshape(-1,1) * 1.
            self.x0 = x_k.reshape(-1,1) * 1.
            self.weights = np.ones((m,), dtype=np.float64)
            self.all_m = np.array([m,])
            r = min(m, nx)
        else:
            # # This is requried for when the number of previous steps is less than self.save_last
            # save_last = min(np.int32(self.all_S.shape[1]/r), self.save_last)
            # Store history for S, Y, and X
            self.all_m = np.hstack((m, self.all_m[:self.save_last]))
            m_total = np.sum(self.all_m)

            ri = -1
            self.all_S = np.hstack((S, self.all_S))
            while self.all_S.shape[1] > m_total:
                self.all_S = np.delete(self.all_S, ri, axis=1)
                
            self.all_Y = np.hstack((Y, self.all_Y))
            while self.all_Y.shape[1] > m_total:
                self.all_Y = np.delete(self.all_Y, ri, axis=1)
            

            self.all_X = np.hstack((x_k.reshape(-1,1), self.all_X))
            while self.all_X.shape[1] > self.save_last + 1:
                self.all_X = np.delete(self.all_X, ri, axis=1)

        
        steps_to_keep2 = self.filter_steps(self.tau)
        steps_to_keep = self.filter_steps2(self.tau, True)

        if not (np.sum(steps_to_keep) == len(steps_to_keep)):
            print('hi')

        # print(steps_to_keep)
        # print(steps_to_keep2)
        # print(np.all(steps_to_keep == steps_to_keep2))
        nd = np.sum(steps_to_keep)

        success = False
        self.n_itr += 1
        msg = 'success'

        if nd == 0:
            msg = f'No valid steps found for tau = {self.tau:.2f}'
            print(msg)
            return B, success, msg




        D = self.all_S[:, steps_to_keep]
        GD = self.hess @ D
        # GD = self.all_Y[:, steps_to_keep]
        BD = B @ D
        
        A =self.all_S.T @ self.hess @ self.all_S
        # Do the Block BFGS update
        try:
            B_new = B - BD @ scipy.linalg.solve(D.T @ BD, np.identity(nd), assume_a='pos') @ BD.T + GD @ scipy.linalg.solve(D.T @ GD, np.identity(nd), assume_a='pos') @ GD.T
            # B_new = B - BD @ np.linalg.pinv(D.T @ BD) @ BD.T + GD @ np.linalg.pinv(D.T @ GD) @ GD.T
            self.B = B_new
            success = True
        except scipy.linalg.LinAlgError:
            msg = 'unable to invert matrix'
            print(msg)
            self.tau += 0.1
            return B, success, msg
        
        return B_new, success, msg