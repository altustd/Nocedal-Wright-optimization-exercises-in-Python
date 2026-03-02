"""
simplex_example_13_1_nocedal_wright.py

Recreates Example 13.1 from Chapter 13 (Linear Programming – The Simplex Method)
in Nocedal & Wright, Numerical Optimization (2nd edition, 2006).

Problem:
    Minimize    -4 x₁ - 2 x₂
    subject to   x₁ +   x₂ + x₃       = 5
                2x₁ + 0.5 x₂     + x₄ = 8
                x₁, x₂, x₃, x₄ ≥ 0

Correct optimal solution: x₁ = 11/3 ≈ 3.6667, x₂ = 4/3 ≈ 1.3333,
                           x₃ = 0, x₄ = 0, objective = -52/3 ≈ -17.3333

This educational implementation follows the revised simplex method steps
shown in the book (basis updates, reduced costs, ratio test, etc.).
"""

import numpy as np

# ────────────────────────────────────────────────
# Problem data
# ────────────────────────────────────────────────
c = np.array([-4.0, -2.0, 0.0, 0.0])           # objective coefficients (cᵀx to minimize)
A = np.array([
    [1.0, 1.0, 1.0, 0.0],                      # constraint 1
    [2.0, 0.5, 0.0, 1.0]                       # constraint 2
])
b = np.array([5.0, 8.0])

n = 4   # total variables (x1,x2,x3,x4)
m = 2   # number of constraints

# Initial basis: slack variables {x₃, x₄} → indices 2 and 3 (0-based)
basis = [2, 3]
nonbasis = [0, 1]

TOL = 1e-10   # small tolerance for floating-point comparisons

print("=== Example 13.1 – Revised Simplex Method (Nocedal & Wright) ===\n")

iteration = 0
while True:
    iteration += 1
    print(f"─ Iteration {iteration} ───────────────────────────────────────────────")

    # Extract current basis / nonbasis matrices and costs
    B = A[:, basis]
    N_mat = A[:, nonbasis]
    c_B = c[basis]
    c_N = c[nonbasis]

    # Basic solution: x_B = B⁻¹ b
    x_B = np.linalg.solve(B, b)

    # Lagrange multipliers: λ = (B⁻ᵀ) c_B
    lambda_ = np.linalg.solve(B.T, c_B)

    # Reduced costs for nonbasic variables: s_N = c_N - Nᵀ λ
    s_N = c_N - N_mat.T @ lambda_

    # Current objective value (nonbasics = 0)
    obj = np.dot(c_B, x_B)

    # ─── Display current tableau state ───
    print(f"  Basis variables     : {[i+1 for i in sorted(basis)]}")
    print(f"  x_B                 : {np.round(x_B, 6)}")
    print(f"  λ (multipliers)     : {np.round(lambda_, 6)}")
    print(f"  Nonbasic variables  : {[i+1 for i in sorted(nonbasis)]}")
    print(f"  Reduced costs s_N   : {np.round(s_N, 6)}")
    print(f"  Objective value     : {obj:.6f}")

    # Optimality check
    if np.all(s_N >= -TOL):
        print("\n>>> Optimal solution found (all reduced costs ≥ 0) <<<\n")
        break

    # Choose entering variable (most negative reduced cost)
    enter_local_idx = np.argmin(s_N)
    enter_var = nonbasis[enter_local_idx]
    enter_col = A[:, enter_var]          # entering column (length m)

    # ─── Ratio test ───
    ratios = np.full(m, np.inf)
    for i in range(m):
        if enter_col[i] > TOL:
            ratios[i] = x_B[i] / enter_col[i]

    if np.all(ratios == np.inf):
        print("  Problem is unbounded (no positive pivot element found).")
        break

    # Select leaving variable (minimum positive ratio)
    leave_row = np.argmin(ratios)
    leave_var = basis[leave_row]

    print(f"  Entering variable   : x{enter_var+1} (reduced cost = {s_N[enter_local_idx]:.6f})")
    print(f"  Leaving  variable   : x{leave_var+1} (ratio = {ratios[leave_row]:.6f})")

    # Pivot: update basis
    basis[leave_row] = enter_var
    nonbasis = [j for j in range(n) if j not in basis]

    print("───────────────────────────────────────────────────────────────────────\n")

# ─── Final solution ───
x_opt = np.zeros(n)
x_opt[basis] = x_B

print("Final Optimal Solution:")
print(f"  x₁ = {x_opt[0]:.6f}")
print(f"  x₂ = {x_opt[1]:.6f}")
print(f"  x₃ = {x_opt[2]:.6f} (slack)")
print(f"  x₄ = {x_opt[3]:.6f} (slack)")
print(f"  Objective value    = {obj:.6f}")
print(f"  (equivalent max 4x₁ + 2x₂ = {-obj:.6f})")