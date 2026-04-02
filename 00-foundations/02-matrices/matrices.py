"""
Matrices — The Language Machine Learning Thinks In
Algorithms in Python — Foundations, Part 2

Demonstrates Python nested lists as matrices and NumPy 2D arrays:
creation, access, transpose, addition, scalar multiplication,
matrix multiplication, and ML-relevant examples.
"""

import numpy as np

# =============================================================================
# Part 1 — Python nested lists as matrices
# =============================================================================

# --- Creation: build a matrix as a list of lists ---
print("=== Python matrix: Creation ===")

matrix = [
    [1, 2, 3],
    [4, 5, 6],
]
print(f"2x3 matrix: {matrix}")

# Zero matrix (3x3)
zeros = [[0 for _ in range(3)] for _ in range(3)]
print(f"3x3 zeros:  {zeros}")

# Identity matrix (3x3) — ones on the diagonal
identity = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
print(f"3x3 identity: {identity}")

# --- Access: read a value by row and column (O(1)) ---
print("\n=== Python matrix: Access ===")

print(f"matrix[0][1] = {matrix[0][1]}")  # Row 0, Col 1 → 2
print(f"matrix[1][2] = {matrix[1][2]}")  # Row 1, Col 2 → 6

# --- Transpose: swap rows and columns (O(m*n)) ---
print("\n=== Python matrix: Transpose ===")

rows = len(matrix)
cols = len(matrix[0])
transposed = [[matrix[r][c] for r in range(rows)] for c in range(cols)]
print(f"Original (2x3): {matrix}")
print(f"Transposed (3x2): {transposed}")

# --- Addition: element-wise sum of two matrices (O(m*n)) ---
print("\n=== Python matrix: Addition ===")

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]
print(f"A = {A}")
print(f"B = {B}")
print(f"A + B = {C}")

# --- Scalar multiplication: scale every element (O(m*n)) ---
print("\n=== Python matrix: Scalar multiplication ===")

scalar = 0.01
scaled = [[scalar * A[i][j] for j in range(2)] for i in range(2)]
print(f"0.01 * A = {scaled}")

# --- Matrix multiplication: the dot product (O(m*n*p)) ---
print("\n=== Python matrix: Matrix multiplication ===")

# (2x3) @ (3x2) → (2x2)
M = [[1, 2, 3],
     [4, 5, 6]]

N = [[7, 8],
     [9, 10],
     [11, 12]]

m_rows, m_cols = len(M), len(M[0])
n_cols = len(N[0])

result = [[0] * n_cols for _ in range(m_rows)]
for i in range(m_rows):
    for j in range(n_cols):
        for k in range(m_cols):
            result[i][j] += M[i][k] * N[k][j]

print(f"M (2x3): {M}")
print(f"N (3x2): {N}")
print(f"M @ N (2x2): {result}")

# =============================================================================
# Part 2 — NumPy 2D arrays
# =============================================================================

print("\n=== NumPy matrix: Creation ===")

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(f"A =\n{A}")
print(f"Shape: {A.shape}, ndim: {A.ndim}, dtype: {A.dtype}")

# Creation shortcuts
print(f"\nnp.zeros((2,3)) =\n{np.zeros((2, 3))}")
print(f"\nnp.ones((2,2)) =\n{np.ones((2, 2))}")
print(f"\nnp.eye(3) =\n{np.eye(3)}")

np.random.seed(42)
print(f"\nnp.random.randn(2,3) =\n{np.random.randn(2, 3).round(2)}")

# --- Vectorised operations ---
print("\n=== NumPy matrix: Operations ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Transpose — zero-cost view, not a copy
print(f"A.T =\n{A.T}")

# Addition
print(f"\nA + B =\n{A + B}")

# Scalar multiplication
print(f"\n0.01 * A =\n{0.01 * A}")

# Matrix multiplication (@ operator)
print(f"\nA @ B =\n{A @ B}")

# Element-wise (Hadamard) product — NOT the same as @
print(f"\nA * B (element-wise) =\n{A * B}")

# Axis aggregation
X = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"\nX =\n{X}")
print(f"Sum along columns (axis=0): {X.sum(axis=0)}")
print(f"Mean along rows (axis=1):   {X.mean(axis=1)}")

# =============================================================================
# Part 3 — ML-relevant examples
# =============================================================================

# --- Forward pass: y = X @ W + b ---
print("\n=== ML example: Forward pass (y = X @ W + b) ===")

np.random.seed(0)
X = np.random.randn(4, 3)       # 4 samples, 3 features
W = np.random.randn(3, 2)       # 3 inputs → 2 outputs
b = np.array([0.1, -0.2])       # bias for each output

output = X @ W + b

print(f"X shape: {X.shape}")
print(f"W shape: {W.shape}")
print(f"b shape: {b.shape}")
print(f"Output shape: {output.shape}")
print(f"Output =\n{output.round(3)}")

# --- 2D rotation ---
print("\n=== ML example: 2D Rotation ===")

theta = np.radians(45)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

points = np.array([[1, 0], [0, 1], [1, 1]])
rotated = points @ R.T

print(f"Rotation matrix (45°):\n{R.round(3)}")
print(f"Original points:\n{points}")
print(f"Rotated points:\n{rotated.round(3)}")

# --- Covariance matrix ---
print("\n=== ML example: Covariance matrix ===")

np.random.seed(1)
# Generate correlated 2D data
mean = [0, 0]
cov_true = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov_true, size=1000)

# Center the data
data_centered = data - data.mean(axis=0)

# Compute covariance: (1/n) * X^T @ X
cov_manual = (1 / len(data_centered)) * (data_centered.T @ data_centered)
cov_numpy = np.cov(data.T, bias=True)

print(f"Manual covariance:\n{cov_manual.round(3)}")
print(f"NumPy covariance:\n{cov_numpy.round(3)}")

# --- Normal equation for linear regression ---
print("\n=== ML example: Normal equation ===")

np.random.seed(42)
n_samples = 100
x = np.random.uniform(0, 10, n_samples)
y = 3 * x + 7 + np.random.randn(n_samples) * 2  # y = 3x + 7 + noise

# Build design matrix: column of ones + feature column
X_design = np.column_stack([np.ones(n_samples), x])

# Normal equation: w = (X^T X)^{-1} X^T y
w = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y

print(f"True weights:      intercept=7, slope=3")
print(f"Recovered weights: intercept={w[0]:.3f}, slope={w[1]:.3f}")
