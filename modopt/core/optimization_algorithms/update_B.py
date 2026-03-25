import numpy as np
from scipy.optimize import minimize

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

class AdaptiveMultiSecant1():
    def __init__(self):
        jax_obj = lambda U, V: jnp.linalg.norm(U @ U.T - V @ V.T, ord='fro')
        _obj = jax.jit(jax_obj)
        self._obj  = lambda U, V: np.float64(_obj(U, V))

        _grad = jax.jit(jax.grad(jax_obj, argnums=[0,1]))
        self._grad  = lambda U, V: np.concatenate([dF.ravel() for dF in _grad(U, V)])
        
        def jax_con1(U, V, B, S, Y):
            B_new = B + U @ U.T - V @ V.T
            return jnp.ravel(B_new @ S - Y)
        _con1 = jax.jit(jax_con1)
        self._con1  = lambda U, V, B, S, Y: np.array(_con1(U, V, B, S, Y))

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

        obj  = lambda x: self._obj(x[:r1*nx].reshape(nx, r1), 
                                   x[r1*nx:].reshape(nx, r2))
        grad = lambda x: self._grad(x[:r1*nx].reshape(nx, r1), 
                                    x[r1*nx:].reshape(nx, r2))

        constraints = [
            {'type': 'eq',
            'fun': lambda x: self._con1(x[:r1*nx].reshape(nx, r1), 
                                        x[r1*nx:].reshape(nx, r2), 
                                        B, S, Y),
            'jac': lambda x: self._jac1(x[:r1*nx].reshape(nx, r1), 
                                        x[r1*nx:].reshape(nx, r2), 
                                        B, S, Y)},
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

class AdaptiveMultiSecant2():
    def __init__(self):
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

        B = 0.5*(1e-6 * np.eye(nx)) + 0.5*B

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

class AdaptiveMultiSecant3():
    def __init__(self):
        self.save_last = 3
        self.p     = 1.0 # 1.0 or 2.0
        self.gamma = 1e-2
        self.all_X = None
        self.all_S = None
        self.all_Y = None
        self.weights = None

        def jax_obj(U, B, S_all, Y_all, beta, weights):
            obj = jnp.linalg.norm(U @ U.T, ord='fro')
            B_new = B + U @ U.T
            c_all = weights*(B_new @ S_all - Y_all)

            penalized_obj = beta * obj + jnp.linalg.norm(c_all, ord='fro')
            return penalized_obj

        _obj = jax.jit(jax_obj)
        self._obj  = lambda U, B, S_all, Y_all, beta, weights: np.float64(_obj(U, B, S_all, Y_all, beta, weights))

        _grad = jax.jit(jax.grad(jax_obj, argnums=[0]))
        self._grad  = lambda U, B, S_all, Y_all, beta, weights: np.asarray(_grad(U, B, S_all, Y_all, beta, weights)[0].ravel())

    def update_B(self, B, S, Y, x_k, r=1):
        if self.all_S is None:
            self.all_S = S * 1.0
            self.all_Y = Y * 1.0
            self.all_X = x_k.reshape(-1,1) * 1.
            self.weights = np.ones((r,), dtype=np.float64)
        else:
            save_last = min(np.int32(self.all_S.shape[1]/r), self.save_last)
            self.all_S = np.hstack((S, self.all_S[:,:save_last*r]))
            self.all_Y = np.hstack((Y, self.all_Y[:,:save_last*r]))
            self.all_X = np.hstack((x_k.reshape(-1,1), self.all_X[:,:save_last]))

            dX = x_k.reshape(-1,1) - self.all_X
            distance = np.linalg.norm(dX, axis=0)
            self.weights = np.repeat(1.0 / (1.0 + distance**self.p), r)
            # self.weights = np.repeat(1.0 / np.exp(distance*self.p), r)

        self.weights = self.gamma + (1-self.gamma) * self.weights
        beta = np.min(self.weights) * 1e-1

        self.weights[:] = 1.0
        beta = 1.0

        nx = S.shape[0]
        x0 = np.ones((r*nx,))

        B = 1e-6 * np.eye(nx)

        obj  = lambda x: self._obj(x.reshape(nx, r), B, self.all_S, self.all_Y, beta, self.weights)
        grad = lambda x: self._grad(x.reshape(nx, r), B, self.all_S, self.all_Y, beta, self.weights)

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
            options=solver_options,
            )

        U = np.reshape(results.x, (nx, r))

        if results['success']:
            B_new = B + U @ U.T
        else:
            B_new = B

        print(f"Success of multi-secant update: {results['success']}")

        return B_new, results['success']