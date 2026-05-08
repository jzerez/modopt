'''Benchmark algorithms on unconstrained CUTEst problems (nx<=100)'''

import numpy as np
from modopt import CUTEstProblem, OpenSQP, BFGS
from modopt.core.optimization_algorithms.hvp_uc import HVPUC
from modopt.core.optimization_algorithms.hvp_uc_2 import HVPUC as HVPUC2
import pycutest
import time
import contextlib
import io
import gc # garbage collector
import pickle
import time 
import matplotlib.pyplot as plt
import os
import traceback
from hist import get_list_hist

import numpy as np
import modopt as mo
import jax.numpy as jnp
import tracemalloc

# https://www.sfu.ca/~ssurjano/rosen.html

n = 1000 # dimension
beta = 10 # coupling strength

jax_obj = lambda v: jnp.sum(beta * (v[1:] - v[:-1]**2)**2 + (1 - v[:-1])**2)
x0 = np.zeros((n,))
prob  = mo.JaxProblem(x0=x0, jax_obj=jax_obj, xl=-np.inf, xu=np.inf)
prob_name = f'rosenbrock_{n}_{beta}'


# TODOs:
# initialize with many hvps at first step, then do normal BFGS
# test damped vs skip. Recover superiority over sqp (damped udates, apply hvp after iter10)

# algs = ['OpenSQP', 'HVPUC', 'HVPUC1', 'HVPUC2', 'nHVPUC1', 'nHVPUC2', 'zHVPUC', ]
# algs = ['OpenSQP', 'BFGS', 'Newton', 'iBFGS-1-n', 'iBFGS-2-n', 'bBFGS-1-n', 'bBFGS-2-n']
# algs = ['OpenSQP', 'BFGS', 'iBFGS-1', 'iBFGS-2', 'iBFGS-4', 'bBFGS-2', 'bBFGS-4', 'AMS1-1', 'AMS1-2', 'AMS1-4', 'AMS3-1', 'AMS3-2' , 'AMS3-4']
# algs = ['OpenSQP', 'AMS1-1', 'AMS1-2', 'iBFGS-1']
algs=  ['OpenSQP', 'BFGS', 'iBFGS-1-h', 'iBFGS-2-h', 'iBFGS-4-h', 'iBFGS-10-h', 'iBFGS-20-h',
        'iBFGS-1-s', 'iBFGS-2-s', 'iBFGS-4-s', 'iBFGS-10-s', 'iBFGS-20-s']

# algs = ['iBFGS-1-h', 'iBFGS-4-h', 'iBFGS-3-s']
performance = {}
history = {}

iteration_categories = ['n_iter', 'n_fev', 'n_gev', 'n_hvpev', 'avg_ls_itr']

t0 = time.time()

plot_on = False
save_figs = False
save_results = True
show_figs = False
max_probs = 999
max_prob_size = 100000
min_prob_size = 100
# normalize_step = False
run_name = 'ibfgs_benchmark_large-50'
save_eval_df = True
save_errs = True
err_hist = {}


run_dir = './hvp_outputs/' + run_name

if save_results or save_figs or save_eval_df:
    try:
        os.mkdir(run_dir)
    except FileExistsError as e:
        print('Directory for run name: ', run_name, 'Already exsists')

if save_eval_df:
    with open(run_dir + '/' + run_name + '_evals.csv', 'w') as f:
        f.write('problem,')
        for alg in algs:
            for category in iteration_categories: 
                f.write(alg + '-' + category + ',')
        f.write('\n')

if save_errs:
    for alg in algs:
        with open(run_dir + '/' + run_name + '_' + alg + '_errors.txt', 'w') as f:
            f.write(f'Error log for {alg}\n')
            f.write('----------------\n')

run_start = False
n_probs_solved = int(0)

maxiter=500
n_probs_solved += 1

if save_eval_df:
    with open(run_dir + '/' + run_name + '_evals.csv', 'a') as f:
        f.write(prob_name + ',')

print(i, prob_name)
if plot_on:
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(f'{prob_name} - \n nx = {prob.nx}')
    fig.tight_layout()
    # Set last ax to be blank for text only
    ax = axs[-1]
    ax.axis('off')

alg_count = -1
sqp_info = []
hvp_info = []
for alg in algs:
    alg_count += 1
    solver = alg
    
    
    print(f'\t{alg} ... ', end='')
    start_time = time.time()
    

    try:
        for i in range(1):
            # with contextlib.redirect_stdout(io.StringIO()):
            

            if solver == 'OpenSQP':
                with contextlib.redirect_stdout(io.StringIO()):
                    options = {'maxiter': maxiter, 'opt_tol': 1.22e-4}
                    results = OpenSQP(prob, **options, recording=False, turn_off_outputs=True).solve()
                    sqp_pass = results['success']
                    if plot_on:
                        # Record final performance metrics: nubmer of iterations, value of objective function, etc.
                        ax.text(0.1, 1.0, f'OPEN SQP\nFinal Objective: {results["objective"]:.2e}\nSuccess: {results["success"]}\nIterations: {results["niter"]}\n x*: {np.round(results["x"], 3)}\n opt: {results["optimality"]}',
                                horizontalalignment='left', verticalalignment='top', fontsize=8
                                )
                sqp_info += [results['nfev'], results['ngev'], results['niter'],]
            elif solver == 'hvpuc-2':
                options = {'maxiter': maxiter, 'opt_tol': 1.22e-4}
                results = HVPUC2(prob, **options, recording=False, turn_off_outputs=True).solve()
                hvp_info += [results['nfev'], results['ngev'], results['niter'],]
                
                if sqp_info[0] != hvp_info[0] or sqp_info[1] != hvp_info[1] or sqp_info[2] != hvp_info[2]:
                    print('sldkjf')
                continue
            else:
                
                solver_params = solver.split('-')
                if len(solver_params) > 1:
                    m = int(solver_params[1])
                else:
                    m = 0
                
                if len(solver_params) > 2:
                    if solver_params[2] == 'h':
                        use_secant = False
                else:
                    use_secant = True
        
                method = solver_params[0]



                options = {'maxiter': maxiter, 'opt_tol': 1.22e-4, 'm': m, 'use_secant': use_secant, 'use_exact_hess':method=='Newton', 'method': method}
                # with contextlib.redirect_stdout(io.StringIO()):
                results = HVPUC(prob, **options, recording=False, turn_off_outputs=True).solve()
                hvp_pass = results['success']
                if plot_on:
                    ax = axs[-1]
                    ax.text(0.1, 0.85 - 0.25*alg_count, f'{alg}\nFinal Objective: {results["objective"]:.2e}\nSuccess: {results["success"]}\nIterations: {results["niter"]}\n x*: {np.round(results["x"], 3)}',
                            horizontalalignment='left', verticalalignment='top', fontsize=8
                        )

                if plot_on:
                    # Plot Objective value of U
                    # ax = axs[0]
                    # ax.plot(results['bk_obj'], label=r'AMS $B_k$ obj', color='red')
                    start_idx = np.diff(results['ams_success'], prepend=1) > 0
                    end_idx = np.diff(results['ams_success'], append=0) < 0

                    if len(end_idx) < len(start_idx):
                        end_idx = np.append(end_idx, True)
                    
                    start_idx = np.arange(len(results['ams_success']))[start_idx]
                    end_idx = np.arange(len(results['ams_success']))[end_idx]

                    # for s, e in zip(start_idx, end_idx):
                    #     ax.axvspan(s, e, color='gray', alpha=0.3)

                    # ax.set_yscale('log')
                    # ax.set_ylabel(r'$B_k$ obj')
                    # ax.set_title(r'Objective value of $U$')

                    # Plot Norm of Hessian approximations error
                    ax = axs[0]
                    ax.plot([np.linalg.norm(bk - hk) for bk, hk in zip(results['approx_hess'], results['true_hess'])], label=alg+r' $B_k$ norm')
                    # ax.plot([np.linalg.norm(bk - hk) for bk, hk in zip(results['qn_hess'], results['true_hess'])], label=r'QN $B_k$ norm')
                    ax.set_yscale('log')
                    ax.set_ylabel(r'$B_k - H_k$ norm (Error Norm)')
                    ax.set_xlabel('Iteration #')
                    ax.set_title(r'Norm of $B_k$ Error')
                    for s, e in zip(start_idx, end_idx):
                        ax.axvspan(s, e, color='gray', alpha=0.3)
                    ax.legend()
                    
                    # Plot Cosine Similarity between Hessian Approximations
                    ax = axs[1]
                    c_ams_true = [np.dot(h_ams.flatten(), h_qn.flatten()) / (np.linalg.norm(h_ams) * np.linalg.norm(h_qn)) for h_ams, h_qn in zip(results['approx_hess'], results['true_hess'])]
                    # ax.plot(c_ams_qn, label=r'QN_HVP-QN')
                    ax.plot(c_ams_true, label=f'{alg}-True')
                    # ax.plot(c_qn_true, label=r'QN-True')
                    ax.set_xlabel('Iteration #')
                    ax.set_title('Cosine Similarity')
                    for s, e in zip(start_idx, end_idx):
                        ax.axvspan(s, e, color='gray', alpha=0.3)
                    ax.legend()

                    # Plot Rank of U at each step:
                    ax = axs[2]
                    ranks = [np.linalg.matrix_rank(U) if not np.any(np.isnan(U)) else 0 for U in results['U']]
                    ax.plot(ranks, label='Rank of U')
                    ax.set_xlabel('Iteration #')
                    ax.set_title('Rank of U')
                    for s, e in zip(start_idx, end_idx):
                        ax.axvspan(s, e, color='gray', alpha=0.3)

                    
                

        opt_time = (time.time() - start_time) / time_loop
        success  = results['success']
        print(success)

        print(f'{results["niter"]} iter')
        print(f'n function evals = {results["nfev"]}')
        print(f'n gradient evals = {results["ngev"]}')
        print(f'n hess evals = {prob._hess_count}')
        print(f'n evals: {results["nfev"] + results["ngev"]}')
        print(f'obj = {results["objective"]:.6e}')
        print(f'x* = {np.round(results["x"], 6)}')
    
        nev         = prob._callback_count
        o_evals     = prob._obj_count
        g_evals     = prob._grad_count
        h_evals     = prob._hess_count
        x           = results['x']
        objective   = prob._compute_objective(results['x'])

        nev     = results['nfev'] + results['ngev']
        o_evals = results['nfev']
        g_evals = results['ngev']
        h_evals = 0
        niter = results['niter']
        hvp_evals = results.get('n_hvp', 0)
        B_k_err = results.get('B_k_err', [])
        B_k_ang = results.get('B_k_ang', [])

        if (not 'OpenSQP' == alg) and (not 'hvpuc-2' in alg):
            B_k_err = [np.linalg.norm(bk - hk) for bk, hk in zip(results['approx_hess'], results['true_hess'])]
            B_k_ang = [np.dot(h_ams.flatten(), h_true.flatten()) / (np.linalg.norm(h_ams) * np.linalg.norm(h_true)) for h_ams, h_true in zip(results['approx_hess'], results['true_hess'])]
        

        if not success:
            if niter == maxiter:
                if save_errs:
                    with open(run_dir + '/' + run_name + '_' + alg + '_errors.txt', 'a') as f:
                        f.write(f'\n ERROR FOR {prob_name}: Iteration Limit Reached\n')
                err_hist[alg, 'max_iter'] = 1 + err_hist.get((alg, 'max_iter'), 0)
            else:
                if not 'err_msg' in results.keys():
                    results['err_msg'] = f'OpenSQP Error. Invalid Objective: {np.isnan(results["objective"]) or np.isinf(results["objective"])}'
                err_hist[alg, results['err_msg']] = 1 + err_hist.get((alg, results['err_msg']), 0)
                if save_errs:
                    with open(run_dir + '/' + run_name + '_' + alg + '_errors.txt', 'a') as f:
                        f.write(f'\n ERROR FOR {prob_name}: {results["err_msg"]}')
    except Exception as e:

        print(False)
        print(f'Error: {e}')
        # raise e

        err_type = type(e).__name__
        val = err_hist.get((alg, err_type), 0)
        err_hist[alg, err_type] = val + 1

        if save_errs:
            with open(run_dir + '/' + run_name + '_' + alg + '_errors.txt', 'a') as f:
                f.write(f'\nERROR FOR {prob_name}: {err_type}\n')
                f.write(traceback.format_exc())

        # Uncomment the line below to raise the exception and see the full traceback for debugging
        # raise e

        success = False
        nev = 1e6
        o_evals = 1e6
        g_evals = 1e6
        h_evals = 1e6
        objective = 1e6
        opt_time = 1e62
        
        feasibility = 1e6
        niter = 1e6
    

    performance[prob.problem_name, alg] = {'time': opt_time,
                                            'success': success,
                                            'nev': nev,
                                            'niter': niter,
                                            'nfev': o_evals,
                                            'ngev': g_evals,
                                            'nhvp': hvp_evals,
                                            'objective': objective,
                                            'B_k err': B_k_err,
                                            'B_k ang': B_k_ang}

    # Write algorithm performacne to df
    if save_eval_df:
        with open(run_dir + '/' + run_name + '_evals.csv', 'a') as f:
            # ['n_iter', 'n_fev', 'n_gev', 'n_hvpev', 'avg_ls_itr']
            avg_line_search = (o_evals + g_evals) * 0.5 / niter
            row = [niter, o_evals, g_evals, hvp_evals, np.round(avg_line_search, 3)]
            row_str = ','.join([str(i) for i in row])
            f.write(row_str)
            f.write(',')

if save_eval_df:
    with open(run_dir + '/' + run_name + '_evals.csv', 'a') as f:
        f.write('\n')

if save_figs: 
    fig.savefig(run_dir + '/' + prob_name + '_' + run_name + '.png')
    if show_figs:
        plt.show(block=True)
    else:
        plt.close(fig)

if save_errs:
    for alg in algs:
        with open(run_dir + '/' + run_name + '_' + alg + '_errors.txt', 'a') as f:
            f.write('\n SUMMARY STATS: \n')
            for key, val in err_hist.items():
                if key[0] == alg:
                    f.write(f'{key[1]}: {val}\n')


print(f'Benchmark complete. Elapsed Time: {(time.time() - t0):.3f}')




if save_results:
    from modopt.benchmarking import plot_performance_profiles
    with open(run_dir + '/hvp_benchmark_' + run_name + '.pkl', 'wb') as f:
        pickle.dump(performance, f)
    plot_performance_profiles(performance, save_figname=run_dir + '/performance_' + run_name + '.pdf', show_plot=show_figs)
