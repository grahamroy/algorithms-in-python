# Matrices — The Language Machine Learning Thinks In

### *Algorithms in Python — Foundations, Part 2*

---

In the previous article we stored data in a single row — a one-dimensional array. Five
temperatures, five positions, one line of numbers. That works when you have one feature.
But the moment you have more than one feature, more than one sample, or more than one
neuron, a single row is not enough. You need rows *and* columns. You need a matrix.

A dataset with 1,000 samples and 10 features is a 1,000 × 10 matrix. A neural network
layer that maps 784 inputs to 256 outputs stores its weights in a 784 × 256 matrix. The
equation ŷ = Xw + b from the arrays article — X is a matrix. The moment you move from
toy examples to real machine learning, matrices are everywhere, and every operation you
perform on data is a matrix operation.

Matrices are where linear algebra enters the picture, and linear algebra is the language
machine learning thinks in. This article builds your fluency.

---

## What is a matrix?

A matrix is a two-dimensional array — values arranged in rows and columns. Its size is
described as m × n, where m is the number of rows and n is the number of columns.

In Python, you can represent a matrix as a list of lists:

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
]
```

Two rows, three columns. That is a 2 × 3 matrix. The first row is `[1, 2, 3]`, the
second is `[4, 5, 6]`. Each inner list is a row, and each position within a row is a
column.

A one-dimensional array — the kind we built in Part 1 — is a special case. It is a
matrix with a single row (a *row vector*) or a single column (a *column vector*). Arrays
and matrices are not different things. Matrices are arrays that grew a second dimension.

---

## Core operations

### 1. Creation — building a matrix from scratch

```python
# A 2x3 matrix of values
matrix = [[1, 2, 3],
          [4, 5, 6]]

# A 3x3 matrix of zeros
zeros = [[0 for _ in range(3)] for _ in range(3)]

# A 3x3 identity matrix — ones on the diagonal
identity = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
```

Creating a matrix requires filling every element, so the cost is O(m × n). The identity
matrix — ones on the diagonal, zeros everywhere else — is particularly important. It is
the matrix equivalent of the number 1: multiply any matrix by the identity and you get
the same matrix back.

In machine learning, creation usually means *initialisation*. Before a neural network
can learn anything, its weight matrices must be filled with starting values. Zeros, small
random numbers, or carefully scaled distributions (Xavier, He) — the choice of
initialisation can determine whether training succeeds or fails.

### 2. Access — reading a value by row and column

```python
matrix[0][1]  # Row 0, Column 1 → 2
matrix[1][2]  # Row 1, Column 2 → 6
```

Access is O(1), just like one-dimensional arrays, but now you supply two indices instead
of one. The first index picks the row, the second picks the column.

In a weight matrix, `W[i][j]` is the connection strength between input neuron i and
output neuron j. Looking up a specific weight is an O(1) operation — the same speed
whether your network has ten neurons or ten million.

### 3. Transpose — swapping rows and columns

```python
rows = len(matrix)
cols = len(matrix[0])
transposed = [[matrix[r][c] for r in range(rows)] for c in range(cols)]
```

The transpose flips a matrix across its diagonal: rows become columns and columns become
rows. A 2 × 3 matrix becomes a 3 × 2 matrix. The element at position (i, j) moves to
position (j, i).

Transposing costs O(m × n) because every element must move. But the operation appears
so often in machine learning that it deserves its own section. The normal equation for
linear regression is w = (XᵀX)⁻¹Xᵀy — two transposes in a single formula. Gradient
computations routinely transpose weight matrices. If you read an ML paper and see a
superscript T, that is a transpose.

### 4. Addition — combining two matrices element by element

```python
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]
# C = [[6, 8], [10, 12]]
```

Two matrices can be added only if they have the same shape. The result is a new matrix
where each element is the sum of the corresponding elements. The cost is O(m × n).

Addition shows up everywhere: adding a bias vector to every row of an output matrix,
accumulating gradients across a batch, combining residual connections in deep networks
(where the input is added to the output of a block, skipping layers entirely).

### 5. Scalar multiplication — scaling every element

```python
scaled = [[0.01 * A[i][j] for j in range(2)] for i in range(2)]
# [[0.01, 0.02], [0.03, 0.04]]
```

Multiply every element by a single number. The cost is O(m × n).

This is how the learning rate works. During training, the gradient tells you the
direction to move each weight. The learning rate — a scalar, typically something small
like 0.01 — tells you how far. The update rule `W = W - lr * gradient` is a scalar
multiplication followed by a matrix subtraction.

### 6. Matrix multiplication — THE operation of machine learning

```python
# (2x3) @ (3x2) → (2x2)
M = [[1, 2, 3],
     [4, 5, 6]]

N = [[7, 8],
     [9, 10],
     [11, 12]]

result = [[0] * 2 for _ in range(2)]
for i in range(2):
    for j in range(2):
        for k in range(3):
            result[i][j] += M[i][k] * N[k][j]
# result = [[58, 64], [139, 154]]
```

Matrix multiplication is not element-wise. Each element of the result is the dot product
of a row from the first matrix and a column from the second. An (m × n) matrix
multiplied by an (n × p) matrix produces an (m × p) matrix. The inner dimensions must
match — both must be n — or the multiplication is undefined.

The cost is O(m × n × p) for the naive algorithm. Three nested loops, each proportional
to one of the dimensions. For square matrices of size n, that is O(n³).

This matters because matrix multiplication is *the* operation of machine learning. Every
forward pass through a neural network layer is a matrix multiplication:

> output = input @ weights

If your input has 4 samples with 3 features (a 4 × 3 matrix) and your weight matrix has
3 inputs mapping to 2 outputs (a 3 × 2 matrix), the result is a 4 × 2 matrix — four
samples, each with two output values. The shape rule (m, n) @ (n, p) → (m, p) is the
key to understanding how data flows through a network. Get comfortable with it, because
you will see it in every architecture you study.

---

## From nested lists to NumPy

The nested-list approach works for understanding, but it falls apart for performance. A
matrix multiplication of two 1,000 × 1,000 matrices requires one billion multiply-add
operations. Python's object overhead and interpreted loops make that painfully slow.

NumPy stores a 2D array as a single contiguous block of memory — just like it does for
1D arrays — with metadata describing the shape. There are no nested lists, no pointers,
no per-element Python objects.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A.shape)  # (2, 3)
print(A.ndim)   # 2
print(A.dtype)  # int64
```

NumPy also gives you creation shortcuts that would be tedious to write by hand:

```python
np.zeros((2, 3))         # 2x3 matrix of zeros
np.ones((2, 2))          # 2x2 matrix of ones
np.eye(3)                # 3x3 identity matrix
np.random.randn(2, 3)    # 2x3 matrix of random normals
```

These are the building blocks of weight initialisation. `np.random.randn(784, 256)`
creates a weight matrix for a layer that maps 784 inputs to 256 outputs, filled with
values drawn from a standard normal distribution. One line of code, one contiguous memory
block, ready for fast arithmetic.

---

## Vectorised matrix operations

With NumPy, the triple-nested loop for matrix multiplication becomes a single operator:

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Transpose — zero-cost view, not a copy
A.T

# Addition
A + B

# Scalar multiplication
0.01 * A

# Matrix multiplication
A @ B       # [[19, 22], [43, 50]]

# Element-wise (Hadamard) product — NOT the same as @
A * B       # [[ 5, 12], [21, 32]]
```

The `@` operator versus the `*` operator is a critical distinction. `A @ B` is matrix
multiplication — the row-by-column dot product that produces a new shape. `A * B` is
element-wise multiplication — each element is multiplied by its counterpart in the same
position, and the shape stays the same. Confusing the two is one of the most common bugs
in ML code.

NumPy also lets you aggregate along specific axes:

```python
X = np.array([[1, 2, 3],
              [4, 5, 6]])

X.sum(axis=0)    # [5, 7, 9]  — sum down each column
X.mean(axis=1)   # [2., 5.]   — mean across each row
```

Axis 0 is "down the rows" (collapsing rows), axis 1 is "across the columns" (collapsing
columns). This is how you compute per-feature statistics (axis=0) or per-sample
statistics (axis=1) in a single call.

---

## Matrices in the machine learning pipeline

**Weight matrices.** A dense layer with n_in inputs and n_out outputs is defined by a
single (n_in × n_out) weight matrix. The forward pass is `output = input @ W + b`. A
deep network with five layers is five matrix multiplications chained together.

**Linear transformations.** Rotation, scaling, projection — all are matrix
multiplications. PCA projects data onto principal component axes via a matrix multiply.
Image transformations in computer vision are matrix operations.

**The covariance matrix.** Given a centred dataset X, the covariance matrix is
(1/n) Xᵀ X. It captures how features vary together — high covariance means they move in
the same direction, low means they are independent. PCA eigendecomposes this matrix to
find the directions of maximum variance.

**The normal equation.** Linear regression's closed-form solution is
w = (XᵀX)⁻¹ Xᵀy. Every piece is a matrix operation: transpose, multiply, invert,
multiply again. If you built linear regression from scratch in Part 1 of the algorithms
series, this is the formula you used.

---

## What comes next

A matrix is two-dimensional. But a colour image is three-dimensional — height, width,
and channels. A batch of images is four-dimensional. A video is five-dimensional. The
moment you work with deep learning, two dimensions are not enough.

In the next article, we will generalise matrices to any number of dimensions: the
**tensor**. Tensors are the native data format of PyTorch and TensorFlow, and
understanding them means understanding how data flows through modern neural networks.

The journey from a flat array to a matrix was one dimension. The next step adds as many
as you need.

---

## The complete code

The full script is on GitHub — grab it here and run it yourself:

[**matrices.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/02-matrices/matrices.py)

Run it with:

```bash
pip install numpy
python matrices.py
```

The full repository for this series: [github.com/grahamroy/algorithms-in-python](https://github.com/grahamroy/algorithms-in-python)
