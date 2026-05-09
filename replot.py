import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import modopt as mo
from modopt.benchmarking import plot_performance_profiles


outputs_dir = './hvp_outputs/'
run_name = 'rosenbrock_1000_10_1'

out_dir = outputs_dir + run_name

with open(out_dir + '/hvp_benchmark_' + run_name + '.pkl', 'rb') as f:
    data = pkl.load(f)

for (problem, alg) in data.keys():
    data[problem, alg]['all_nev'] = data[problem, alg]['nfev'] + data[problem, alg]['ngev'] + data[problem, alg]['nhvp']
    print(f'{problem}, {alg} nev: {data[problem, alg]["nev"]}')
    print(f'{problem}, {alg} nhvp: {data[problem, alg]["nhvp"]}')
    print(f'{problem}, {alg} niter: {data[problem, alg]["niter"]}')
    print(f'{problem}, {alg} all_nev: {data[problem, alg]["all_nev"]}')
    B_ks = data[problem, alg]["B_ks"]
    if B_ks:
        print(f'{data[problem, alg]["B_ks"][0]}')
    else:
        print(f'No B_ks')
    print('-'*50)

# Performance plots
# plot_performance_profiles(data, save_figname=out_dir + '/performance_' + run_name + '.pdf', show_plot=True, fields=['nev', 'niter', 'all_nev'])
# plot_performance_profiles(data, save_figname=out_dir + '/performance_' + run_name + '.pdf', show_plot=True, fields=['nhvp'], alg_blacklist=['BFGS', 'OpenSQP'])

# Error plots
algs = np.unique([alg for (prob, alg) in data.keys()])
probs = np.unique([prob for (prob, alg) in data.keys()])

for prob in probs:
    for alg in algs:
        fig, ax = plt.subplots()
        ax.plot(data[prob, alg]['nevs'], data[prob,alg]['B_k_errs'])
        
