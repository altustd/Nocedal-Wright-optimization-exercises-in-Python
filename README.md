# Nocedal-Wright-optimization-exercises-in-Python

Python implementations and demonstrations of selected exercises and algorithms from the classic textbook on **Numerical Optimization**.

This repository implements various optimization algorithms and solves exercises from *Numerical Optimization* (2nd edition) using Python. The focus is on understanding core numerical optimization techniques—such as line search methods, Newton's method, quasi-Newton (e.g., BFGS), trust-region approaches, conjugate gradients, and constrained optimization—by coding them from scratch where appropriate, while also comparing results against Python's built-in tools like `scipy.optimize`.

## Attribution / Bibliography

This project draws inspiration from exercises, algorithms, and concepts in the following source on **Numerical Optimization**. All code implementations are original (or adapted for educational clarity) and do not reproduce protected material from the book.

Nocedal, Jorge, and Stephen J. Wright. *Numerical Optimization*. 2nd ed., Springer, 2006.

(ISBN-13: 978-0387303031 | Publisher page: https://link.springer.com/book/9780387303031)

## Project Goals
- Demonstrate key **numerical optimization** methods from the book in Python.
- Implement algorithms manually (e.g., steepest descent, BFGS updates, Armijo line search) to build intuition.
- Use Python's optimization libraries (primarily `scipy.optimize`, NumPy, Matplotlib) for validation, benchmarking, and visualization.
- Cover topics like unconstrained minimization, nonlinear least squares, equality/inequality constraints, and more.
- Educational/non-commercial use only—refer to the book's copyright for details.

## Usage Notes
- See the notebooks (.ipynb) or .py files organized by chapter/exercise (e.g., Chapter 3: Line Search, Chapter 6: Quasi-Newton, Chapter 12: Interior-Point Methods).
- Requirements: Python 3.x + NumPy, SciPy, Matplotlib (install via `pip install numpy scipy matplotlib` or use a conda env).
- Many implementations test classic benchmark functions (Rosenbrock, Himmelblau, etc.) from the literature.

## License
MIT License (or your preference—add a LICENSE file)

Happy optimizing! If you're following along with the book, these implementations should help solidify the theory through code.
