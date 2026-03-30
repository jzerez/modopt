import numpy as np
from scipy.optimize import minimize

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

"""
Different Strategies for updating the hessian approximation using HVP information.
"""


# This Method is equivalent to Direction 3 from 2-19 meeting notes
class AdaptiveMultiSecant1():

    # Here, we're framing the update as an optimization problem
    # Want to find matrices U and V with the "least-square magnitude"
    # That satisfy:
    #   Our secant conditions
    #   a positive definite hessian update 
    def __init__(self):
        jax_obj = lambda U, V: jnp.linalg.norm(U @ U.T - V @ V.T, ord='fro')
        _obj = jax.jit(jax_obj)
        self._obj  = lambda U, V: np.float64(_obj(U, V))

        _grad = jax.jit(jax.grad(jax_obj, argnums=[0,1]))
        self._grad  = lambda U, V: np.concatenate([dF.ravel() for dF in _grad(U, V)])
        
        # Multi-secant condition
        def jax_con1(U, V, B, S, Y):
            B_new = B + U @ U.T - V @ V.T
            return jnp.ravel(B_new @ S - Y)
        _con1 = jax.jit(jax_con1)
        self._con1  = lambda U, V, B, S, Y: np.array(_con1(U, V, B, S, Y))

        # Forcing positive-definiteness (but in a weird way)? 
        def jax_con2(U, V, B):
            B_new = B + U @ U.T - V @ V.T
            w = jnp.real(jnp.linalg.eigvals(B_new))
            return jnp.array([jnp.min(w) - 1e-6])
        _con2 = jax.jit(jax_con2)
        self._con2  = lambda U, V, B: np.array(_con2(U, V, B))

        _jac1 = jax.jit(jax.jacobian(jax_con1, argnums=[0,1]))
        def temp_jac1(U, V, B, S, Y):
            dU, dV = _jac1(U, V, B, S, Y)
            return np.hstack((dU.reshape(dU.shape[0], -1), dV.reshape(dV.shape[0], -1)))
        self._jac1 = temp_jac1

        _jac2 = jax.jit(jax.jacobian(jax_con2, argnums=[0,1]))
        def temp_jac2(U, V, B):
            dU, dV = _jac2(U, V, B)
            return np.hstack((dU.reshape(dU.shape[0], -1), dV.reshape(dV.shape[0], -1)))
        self._jac2 = temp_jac2

    def update_B(self, B, S, Y, r1=1, r2=1):
        nx = B.shape[0]
        x0 = np.ones(((r1+r2)*nx,))

        # Unpack the design vector (x0) into U components and V components.
        # U is comprised of r1 vectors of length nx
        # V is comprised of r2 vectors of length nx

        obj  = lambda x: self._obj(x[:r1*nx].reshape(nx, r1), 
                                   x[r1*nx:].reshape(nx, r2))
        grad = lambda x: self._grad(x[:r1*nx].reshape(nx, r1), 
                                    x[r1*nx:].reshape(nx, r2))

        constraints = [
            # Enforce secant condition
            {'type': 'eq',
            'fun': lambda x: self._con1(x[:r1*nx].reshape(nx, r1), 
                                        x[r1*nx:].reshape(nx, r2), 
                                        B, S, Y),
            'jac': lambda x: self._jac1(x[:r1*nx].reshape(nx, r1), 
                                        x[r1*nx:].reshape(nx, r2), 
                                        B, S, Y)},
            
            # Enforce positive definiteness
            {'type': 'ineq',
            'fun': lambda x: self._con2(x[:r1*nx].reshape(nx, r1), 
                                        x[r1*nx:].reshape(nx, r2), 
                                        B),
            'jac': lambda x: self._jac2(x[:r1*nx].reshape(nx, r1), 
                                        x[r1*nx:].reshape(nx, r2), 
                                        B)},
        ]

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

        U = np.reshape(results.x[:nx*r1], (nx, r1))
        V = np.reshape(results.x[nx*r1:], (nx, r2))

        if results['success']:
            B_new = B + U @ U.T - V @ V.T
        else:
            B_new = B

        print(f"Success of multi-secant update: {results['success']}")

        return B_new, results['success']

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
        self.save_last = 5
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
            r = 2
        else:
            # # This is requried for when the number of previous steps is less than self.save_last
            # save_last = min(np.int32(self.all_S.shape[1]/r), self.save_last)
            # Store history for S, Y, and X
            self.all_m = np.hstack((m, self.all_m[:self.save_last]))
            m_total = np.sum(self.all_m)

            self.all_S = np.hstack((S, self.all_S))
            while self.all_S.shape[1] > m_total:
                self.all_S = np.delete(self.all_S, -2, axis=1)
                
            self.all_Y = np.hstack((Y, self.all_Y))
            while self.all_Y.shape[1] > m_total:
                self.all_Y = np.delete(self.all_Y, -2, axis=1)
            

            self.all_X = np.hstack((x_k.reshape(-1,1), self.all_X))
            while self.all_X.shape[1] > self.save_last + 1:
                self.all_X = np.delete(self.all_X, -2, axis=1)

            dX = x_k.reshape(-1,1) - self.all_X
            distance = np.linalg.norm(dX, axis=0)
            self.weights = np.repeat(1.0 / (1.0 + distance**self.p), self.all_m)
            r = max(min(m_total, nx), 2)
            # self.weights = np.repeat(1.0 / np.exp(distance*self.p), r)
        if self.n_itr > 100:
            print(self.weights)
        # Problem with the weights. "save last" is ambiguous. It should refer to the number of 
        # previous HVPs. But this breaks down when the number of HVPs is not consistent between steps. 
        # We will run into problems when nx > 6. The first step will use at least 6 HVPs, rather than 3. 
        self.weights = self.gamma + (1-self.gamma) * self.weights

        # Beta is how much we care about the norm of UU^T. 
        # Here we are saying that the update for B should be dominated by
        # satisfying a weighted sum of past secant conditions.
        beta = np.min(self.weights) * 0.10
        
        # Seed optimizer with previous U matrix, if available. 
        if not (self.prev_U is None) and (self.all_S.shape[1] > self.save_last):
            x0 = np.reshape(self.prev_U, (r*nx))
        else:
            x0 = np.ones((r*nx,))

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