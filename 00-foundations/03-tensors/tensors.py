"""
Tensors — The Native Data Format of Deep Learning
Algorithms in Python — Foundations, Part 3

Demonstrates tensors using NumPy:
creation, indexing, reshaping, broadcasting,
and deep-learning-relevant examples (batched images, conv filters, sequence data).
"""

import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# Part 1 — Tensors as generalised arrays (NumPy)
# =============================================================================

print("=== Tensor ranks ===")

# Rank 0 — scalar
scalar = np.array(3.14)
print(f"Rank 0 (scalar):  value={scalar},  ndim={scalar.ndim},  shape={scalar.shape}")

# Rank 1 — vector (1D array — covered in Part 1)
vector = np.array([1.0, 2.0, 3.0])
print(f"Rank 1 (vector):  value={vector},  ndim={vector.ndim},  shape={vector.shape}")

# Rank 2 — matrix (2D array — covered in Part 2)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"Rank 2 (matrix):  ndim={matrix.ndim},  shape={matrix.shape}")

# Rank 3 — 3D tensor (e.g. single colour image: height × width × channels)
image = np.random.randint(0, 256, size=(28, 28, 3), dtype=np.uint8)
print(f"Rank 3 (image):   ndim={image.ndim},  shape={image.shape}  (H×W×C)")

# Rank 4 — 4D tensor (batch of images: batch × height × width × channels)
batch = np.random.randint(0, 256, size=(32, 28, 28, 3), dtype=np.uint8)
print(f"Rank 4 (batch):   ndim={batch.ndim},  shape={batch.shape}  (N×H×W×C)")

# Rank 5 — 5D tensor (video: batch × frames × height × width × channels)
video = np.zeros((4, 16, 64, 64, 3), dtype=np.uint8)
print(f"Rank 5 (video):   ndim={video.ndim},  shape={video.shape}  (N×T×H×W×C)")

# =============================================================================
# Part 2 — Creation
# =============================================================================

print("\n=== Creation ===")

# From nested lists
t = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
print(f"From nested lists — shape: {t.shape}, ndim: {t.ndim}")

# Zeros, ones, random
print(f"np.zeros((2,3,4)) shape: {np.zeros((2, 3, 4)).shape}")
print(f"np.ones((3,3,3))  shape: {np.ones((3, 3, 3)).shape}")

np.random.seed(42)
rand_tensor = np.random.randn(2, 3, 4)
print(f"np.random.randn(2,3,4) shape: {rand_tensor.shape}")

# =============================================================================
# Part 3 — Indexing and slicing
# =============================================================================

print("\n=== Indexing ===")

# t has shape (2, 2, 2)
print(f"t =\n{t}")
print(f"t[0]          = {t[0]}        # first 2D slice")
print(f"t[0][1]       = {t[0][1]}     # second row of first slice")
print(f"t[0][1][0]    = {t[0][1][0]}  # single element: depth=0, row=1, col=0  (3)")
print(f"t[0, 1, 0]    = {t[0, 1, 0]}  # same, cleaner syntax")

# Slicing
print(f"\nt[:, :, 0] =\n{t[:, :, 0]}   # all elements at column 0 across all slices")
print(f"t[1, :, :]  =\n{t[1, :, :]}   # full second depth slice")

# =============================================================================
# Part 4 — Reshaping and squeezing
# =============================================================================

print("\n=== Reshaping ===")

flat = np.arange(24)
print(f"Original (flat): {flat.shape}")

# Reshape to 3D
cube = flat.reshape(2, 3, 4)
print(f"reshape(2,3,4): {cube.shape}")

# Reshape to 4D
four_d = flat.reshape(2, 2, 2, 3)
print(f"reshape(2,2,2,3): {four_d.shape}")

# -1 lets NumPy infer one dimension
auto = flat.reshape(4, -1)
print(f"reshape(4,-1): {auto.shape}  # NumPy infers 6")

# Flatten back to 1D
back_flat = cube.flatten()
print(f"flatten(): {back_flat.shape}")

# squeeze/expand_dims
single_batch = np.random.randn(1, 28, 28, 3)
squeezed = np.squeeze(single_batch, axis=0)
print(f"\nsqueeze (1,28,28,3) -> {squeezed.shape}")

expanded = np.expand_dims(squeezed, axis=0)
print(f"expand_dims (28,28,3) -> {expanded.shape}")

# =============================================================================
# Part 5 — Broadcasting
# =============================================================================

print("\n=== Broadcasting ===")

# Add a bias (shape (3,)) to a batch of outputs (shape (4, 3))
outputs = np.random.randn(4, 3)
bias    = np.array([0.1, -0.2, 0.3])

result = outputs + bias   # bias is broadcast across all 4 rows
print(f"outputs shape: {outputs.shape}")
print(f"bias shape:    {bias.shape}")
print(f"result shape:  {result.shape}  (bias added to every sample)")

# Normalise each channel in a batch of images
# images: (N, H, W, C), mean/std: (C,)
images = np.random.randint(0, 256, (8, 32, 32, 3)).astype(float)
mean   = images.mean(axis=(0, 1, 2))   # mean per channel — shape (3,)
std    = images.std(axis=(0, 1, 2))    # std per channel  — shape (3,)
normalised = (images - mean) / std     # broadcast over N, H, W
print(f"\nimages shape:     {images.shape}")
print(f"mean/std shape:   {mean.shape}")
print(f"normalised shape: {normalised.shape}  (channel stats broadcast over N,H,W)")

# =============================================================================
# Part 6 — ML-relevant examples
# =============================================================================

# --- Example 1: Batched matrix multiplication (linear layer on a batch) ---
print("\n=== ML example: Batched linear layer ===")

np.random.seed(0)
batch_size, seq_len, d_model = 4, 10, 8
X = np.random.randn(batch_size, seq_len, d_model)   # (4, 10, 8) — batch of sequences
W = np.random.randn(d_model, 4)                      # (8, 4) — weight matrix

# Apply W to every (seq_len, d_model) slice in the batch
# np.matmul broadcasts: (..., n, m) @ (m, p) -> (..., n, p)
output = X @ W                                       # (4, 10, 4)
print(f"Input   X:  {X.shape}  (batch × seq_len × d_model)")
print(f"Weights W:  {W.shape}  (d_model × d_out)")
print(f"Output X@W: {output.shape}  (batch × seq_len × d_out)")

# --- Example 2: CNN weight tensor ---
print("\n=== ML example: CNN filter tensor ===")

# A Conv2D layer: out_channels × in_channels × kernel_h × kernel_w
conv_weights = np.random.randn(64, 3, 3, 3)   # 64 filters, each 3×3 on RGB
print(f"Conv2D weights shape: {conv_weights.shape}")
print(f"  64 filters | 3 input channels | 3×3 spatial kernel")

# One filter, one channel
single_filter = conv_weights[0, 0, :, :]
print(f"  First filter, first channel:\n{single_filter.round(3)}")

# --- Example 3: Sequence / Transformer data ---
print("\n=== ML example: Transformer input tensor ===")

batch_size, seq_len, embed_dim = 8, 50, 512
tokens = np.random.randn(batch_size, seq_len, embed_dim)
print(f"Token tensor shape: {tokens.shape}")
print(f"  {batch_size} sequences | {seq_len} tokens each | {embed_dim}-dim embeddings")

# --- Example 4: Reduction along axes ---
print("\n=== ML example: Axis reductions ===")

activations = np.random.randn(32, 10)   # 32 samples, 10 class logits
class_scores = activations.mean(axis=0)   # mean per class across batch
sample_norms = np.linalg.norm(activations, axis=1)   # L2 norm per sample

print(f"Activations:   {activations.shape}")
print(f"Mean per class (axis=0): {class_scores.shape}  — {class_scores.round(3)}")
print(f"Norm per sample (axis=1): {sample_norms.shape}  — {sample_norms.round(3)}")
