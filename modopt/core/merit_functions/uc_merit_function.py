import numpy as np
from modopt import MeritFunction


# Note: Augmented Lagrangian is a function of  x, lag_mult and slack variables
# Note: This Merit function is for problems with constraints: c_e(x) = 0, c_i(x) >= 0
class UCMerit(MeritFunction):
    """
    Merit function with no constraintx.

    Parameters
    ----------
    f : callable
        Objective function.
    g : callable
        Objective gradient.
    nx : int
        Number of optimization/design variables.
    
    Attributes
    ----------
    cache : dict
        Dictionary to store the latest function evaluations for 'f', 'c', 'g' and 'j'.
        Keys are the function names 'f', 'c', 'g' and 'j', 
        and values are tuples of the form (x, f(x)).
    eval_count : dict
        Dictionary to store the number of times each function has been evaluated.
        Keys are the function names 'f', 'c', 'g' and 'j',
        and values are the number of evaluations.
    """
    def initialize(self):
        pass

    def setup(self):
        pass

    def compute_function(self, v):
        nx   = self.options['nx']

        x = v[:nx]

        self.update_functions_in_cache(['f'], x)
        obj = self.cache['f'][1]
        # obj = self.options['f'](x)

        return obj

        # nbi  = self.options['non_bound_indices']
        # return obj - np.dot(lag_mult[nbi], (con - slacks)[nbi]) + 0.5 * np.inner(rho[nbi], (con - slacks)[nbi]**2)

# Note: Gradient is evaluated with respect to x, lag_mult and slack variables
    def compute_gradient(self, v):
        nx = self.options['nx']
        x = v[:nx]

        self.update_functions_in_cache(['g'], x)
        grad = self.cache['g'][1]
        # grad = self.options['g'](x)

        grad_x = grad

        # nbi  = self.options['non_bound_indices']
        # grad_x = grad - jac[nbi].T @ (lag_mult[nbi] - (rho[nbi]* (con[nbi] - slacks[nbi])))

        # print('compute_gradient', np.concatenate((grad_x, grad_lag_mult, grad_slacks)))

        return grad_x