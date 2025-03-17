# Conjugate-gradients
Conjugate Gradient Method for Large Sparse Systems

This project implements the Conjugate Gradient (CG) method (without preconditioning) to solve large sparse linear systems Ax = b.

Features
✅ Custom CG Solver: Computes x and tracks error at each iteration (‖Axᵢ - b‖₂)
✅ Sparse Matrix Handling: Runs on a 10,000 × 10,000 sparse matrix A
✅ Pickle File Input: Reads A, x₀, and b from a provided .pkl file
✅ Error Analysis: Plots log₁₀(error) vs. iteration count

Usage
Load the provided pickle file containing A, x₀, and b.
Run the CG solver to compute x and track convergence.
Generate a plot of log₁₀(error) vs. iteration.
This project explores iterative methods for solving large sparse systems, demonstrating convergence behavior and numerical stability of the Conjugate Gradient algorithm.
