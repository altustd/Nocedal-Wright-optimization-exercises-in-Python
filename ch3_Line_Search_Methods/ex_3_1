import numpy as np

def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_rosenbrock(x):
    u = x[1] - x[0]**2
    g0 = -400 * x[0] * u - 2 * (1 - x[0])
    g1 = 200 * u
    return np.array([g0, g1])

def hess_rosenbrock(x):
    h11 = -400 * (x[1] - 3 * x[0]**2) + 2
    h12 = -400 * x[0]
    h22 = 200
    return np.array([[h11, h12], [h12, h22]])

def backtracking_line_search(x, p, grad, f, alpha0=1.0, rho=0.5, c=1e-4, max_iter=100):
    alpha = alpha0
    fx = f(x)
    gradp = np.dot(grad, p)
    for _ in range(max_iter):
        if f(x + alpha * p) <= fx + c * alpha * gradp:
            return alpha
        alpha *= rho
    raise ValueError("Line search did not converge")

def optimize(method, x0, tol=1e-8, max_iter=100000):
    x = np.array(x0, dtype=float)
    print(f"\nOptimizing with {method} method from starting point {x}")
    iter_count = 0
    while iter_count < max_iter:
        grad = grad_rosenbrock(x)
        norm_grad = np.linalg.norm(grad)
        if norm_grad < tol:
            break
        if method == "steepest":
            p = -grad
        elif method == "newton":
            hess = hess_rosenbrock(x)
            p = np.linalg.solve(hess, -grad)
        else:
            raise ValueError("Unknown method")
        alpha = backtracking_line_search(x, p, grad, rosenbrock)
        print(f"Iteration {iter_count + 1}: step length {alpha}")
        x += alpha * p
        iter_count += 1
    print(f"Converged after {iter_count} iterations")
    print(f"Final point: {x}")
    print(f"Final function value: {rosenbrock(x)}")
    print(f"Final gradient norm: {norm_grad}")
    return x

# First starting point
print("First starting point (1.2, 1.2)")
optimize("steepest", [1.2, 1.2])
optimize("newton", [1.2, 1.2])

# Second starting point
print("\nSecond starting point (-1.2, 1)")
optimize("steepest", [-1.2, 1])
optimize("newton", [-1.2, 1])
