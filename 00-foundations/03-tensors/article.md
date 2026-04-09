# Tensors — The Native Data Format of Deep Learning

### *Algorithms in Python — Foundations, Part 3*

---

In Part 1 we stored numbers in a single row — a one-dimensional array. In Part 2 we
grew that row into a table — a two-dimensional matrix. A matrix handles a dataset with
multiple features, a layer of weights, a covariance structure. But what about a colour
photograph?

A photograph is not a table. It has height, width, *and* channels — three dimensions.
A batch of photographs has four. A video has five. The moment you work with images,
sequences, or any real deep learning model, two dimensions are not enough. You need a
data structure that generalises to any number of dimensions.

That structure is the **tensor**.

---

## What is a tensor?

A tensor is an n-dimensional array. Every data structure we have seen so far is a
special case:

- A **scalar** is a rank-0 tensor — a single number, shape `()`
- A **vector** is a rank-1 tensor — a 1D array, shape `(n,)`
- A **matrix** is a rank-2 tensor — a 2D array, shape `(m, n)`
- An **image** is a rank-3 tensor — height × width × channels, shape `(H, W, C)`
- A **batch of images** is a rank-4 tensor — shape `(N, H, W, C)`
- A **video** is a rank-5 tensor — batch × frames × height × width × channels

The rank is just the number of dimensions. Nothing fundamentally changes when you go
from 2 to 3 to 5 — the structure is the same, there are just more axes to track.

```python
import numpy as np

# Rank 0 — scalar
scalar = np.array(3.14)
print(f"Rank 0: shape={scalar.shape}, ndim={scalar.ndim}")   # shape=(), ndim=0

# Rank 3 — colour image (28×28 pixels, 3 channels)
image = np.random.randint(0, 256, size=(28, 28, 3), dtype=np.uint8)
print(f"Rank 3: shape={image.shape}, ndim={image.ndim}")      # shape=(28, 28, 3), ndim=3

# Rank 4 — batch of 32 colour images
batch = np.random.randint(0, 256, size=(32, 28, 28, 3), dtype=np.uint8)
print(f"Rank 4: shape={batch.shape}, ndim={batch.ndim}")      # shape=(32, 28, 28, 3), ndim=4
```

In NumPy, every `np.array` is already a tensor — the library does not have separate
types for "matrix" and "tensor". A 2D array has `ndim=2`, a 4D array has `ndim=4`.
The same object, the same methods, different shapes.

---

## Creation

Creating tensors follows the same patterns as arrays and matrices, extended to
arbitrary depth:

```python
# From nested lists
t = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
print(t.shape)   # (2, 2, 2)

# Zeros, ones, random — pass a tuple of dimensions
np.zeros((2, 3, 4))          # shape (2, 3, 4)
np.ones((3, 3, 3))           # shape (3, 3, 3)
np.random.randn(2, 3, 4)     # shape (2, 3, 4), values from N(0,1)
```

The cost is O(total elements) — you are filling every cell. A tensor of shape
`(32, 28, 28, 3)` has 32 × 28 × 28 × 3 = 75,264 elements.

---

## Indexing and slicing

Indexing extends naturally from matrices: one index per dimension.

```python
t = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])   # shape (2, 2, 2)

t[0]           # first slice along axis 0: [[1,2],[3,4]]
t[0][1]        # second row of first slice: [3, 4]
t[0, 1, 0]     # single element at depth=0, row=1, col=0: 3
```

The comma syntax `t[0, 1, 0]` is cleaner than chained brackets and works for any
number of dimensions. In deep learning code you will see it constantly.

Slicing works across any axis:

```python
t[:, :, 0]    # all elements at position 0 along the last axis — shape (2, 2)
t[1, :, :]    # full second depth slice — shape (2, 2)
```

This is how you extract a single channel from an image (`image[:, :, 0]`), or a
single sample from a batch (`batch[i, :, :, :]`).

---

## Reshaping

Reshaping is one of the most frequent operations in deep learning code. A tensor's
data never changes — only how it is indexed.

```python
flat = np.arange(24)          # shape (24,)

cube    = flat.reshape(2, 3, 4)      # shape (2, 3, 4)
four_d  = flat.reshape(2, 2, 2, 3)  # shape (2, 2, 2, 3)
auto    = flat.reshape(4, -1)        # shape (4, 6)  — NumPy infers the 6

back    = cube.flatten()             # shape (24,) — back to 1D
```

The `-1` is shorthand for "figure it out". If you have 24 elements and you ask for
shape `(4, -1)`, NumPy fills in 6 because 4 × 6 = 24. This is particularly useful
when the batch size is variable: `x.reshape(batch_size, -1)` flattens everything
except the first dimension.

Two operations appear constantly in model code:

```python
# squeeze — remove a dimension of size 1
single = np.random.randn(1, 28, 28, 3)   # shape (1, 28, 28, 3)
squeezed = np.squeeze(single, axis=0)     # shape (28, 28, 3)

# expand_dims — add a dimension of size 1
expanded = np.expand_dims(squeezed, axis=0)   # shape (1, 28, 28, 3)
```

`squeeze` removes a batch dimension when you are processing a single sample.
`expand_dims` adds one back when a function expects a batch. The data is unchanged;
only the shape changes.

Reshaping is O(1) in NumPy — it just updates the metadata. The underlying memory
block is the same. This is what makes it cheap even on large tensors.

---

## Broadcasting

Broadcasting is NumPy's rule for performing operations on tensors of different shapes.
Instead of requiring identical shapes, it *expands* the smaller tensor to match the
larger one — without copying data.

The rule: dimensions are aligned from the right. A dimension of size 1 is stretched
to match whatever it is paired with.

```python
outputs = np.random.randn(4, 3)    # shape (4, 3) — 4 samples, 3 outputs
bias    = np.array([0.1, -0.2, 0.3])   # shape (3,)

result = outputs + bias            # shape (4, 3) — bias added to every row
```

NumPy aligns `(4, 3)` and `(3,)` from the right: `3` matches `3`, and the missing
first dimension is treated as 1 and stretched to 4. The result is as if you had
copied the bias four times and added it element-wise — but no copy actually happens.

A more realistic example — normalising image batches:

```python
images = np.random.randint(0, 256, (8, 32, 32, 3)).astype(float)
mean   = images.mean(axis=(0, 1, 2))   # shape (3,) — one mean per channel
std    = images.std(axis=(0, 1, 2))    # shape (3,)

normalised = (images - mean) / std     # shape (8, 32, 32, 3)
```

NumPy broadcasts `(3,)` across `(8, 32, 32, 3)`. Each channel is normalised by
its own mean and standard deviation across the entire batch. One line of code,
no loops, no copies.

Broadcasting rules break down to: align shapes right-to-right; a missing dimension
is treated as 1; a dimension of 1 can always be stretched to match.

---

## Tensors in the machine learning pipeline

**Batched forward pass.** Neural network code rarely processes one sample at a time.
It processes *batches*. The input to a linear layer is not a vector — it is a matrix
(batch × features). The input to a convolutional layer is not an image — it is a 4D
tensor (batch × height × width × channels).

NumPy's `@` operator handles this naturally:

```python
batch_size, seq_len, d_model = 4, 10, 8
X = np.random.randn(batch_size, seq_len, d_model)   # (4, 10, 8)
W = np.random.randn(d_model, 4)                      # (8, 4)

output = X @ W    # (4, 10, 4) — the same W applied to every element of the batch
```

`np.matmul` (which `@` calls) broadcasts over all but the last two dimensions. The
weight matrix `W` is automatically applied to every `(seq_len, d_model)` slice in
the batch.

**CNN weight tensors.** A convolutional layer's weights are a 4D tensor:
`(out_channels, in_channels, kernel_height, kernel_width)`.

```python
conv_weights = np.random.randn(64, 3, 3, 3)
# 64 output channels
# 3 input channels (RGB)
# 3×3 spatial kernel
```

Training a CNN means updating these 64 × 3 × 3 × 3 = 1,728 numbers so that the
learned filters respond to meaningful patterns — edges, curves, textures. The tensor
shape encodes the architecture.

**Transformer token tensors.** Language models represent text as a 3D tensor:

```python
tokens = np.random.randn(8, 50, 512)
# 8 sequences in the batch
# 50 tokens per sequence
# 512-dimensional embedding per token
```

Every operation in a transformer — attention, feed-forward layers, layer normalisation
— is a tensor operation on this shape. The batch, sequence, and embedding dimensions
each have a precise meaning that the architecture preserves throughout.

**Reductions across axes.** The `axis` parameter lets you aggregate along any
dimension:

```python
activations = np.random.randn(32, 10)   # 32 samples, 10 class logits

class_scores = activations.mean(axis=0)             # shape (10,) — average per class
sample_norms = np.linalg.norm(activations, axis=1)  # shape (32,) — L2 norm per sample
```

Choosing the right axis is one of the things that trips people up most in ML code.
`axis=0` collapses the batch dimension (result: one value per class). `axis=1`
collapses the class dimension (result: one value per sample).

---

## From NumPy to PyTorch

Everything in this article has used NumPy. In practice, deep learning uses
**PyTorch** (or TensorFlow). PyTorch's `torch.Tensor` is almost identical to
`np.ndarray`, with one critical addition: it can run on a GPU and it tracks
operations for automatic differentiation.

The core ideas carry directly:

| NumPy | PyTorch |
|-------|---------|
| `np.array([...])` | `torch.tensor([...])` |
| `x.shape` | `x.shape` |
| `x.reshape(...)` | `x.reshape(...)` or `x.view(...)` |
| `x @ y` | `x @ y` or `torch.matmul(x, y)` |
| `np.squeeze(x)` | `x.squeeze()` |
| `np.expand_dims(x, 0)` | `x.unsqueeze(0)` |

Understanding NumPy tensors first means you already understand 90% of PyTorch's
API. The differences — `.cuda()`, `.grad`, `autograd` — are the additions that make
the GPU and backpropagation work.

---

## What comes next

You have now seen all three fundamental data structures:

- **Arrays** — one dimension, fast sequential access, the basis of all numerical
  computation
- **Matrices** — two dimensions, the workhorse of linear algebra, weight matrices
  and data tables
- **Tensors** — n dimensions, the native format of deep learning, images, sequences,
  batches

The next article moves from how data is stored to how data is *processed*. We will
look at **Linked Lists** — a different kind of structure that prioritises flexible
insertion over fast random access, and that underpins streaming data pipelines and
memory management.

---

## The complete code

The full script is on GitHub — grab it here and run it yourself:

[**tensors.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/03-tensors/tensors.py)

Run it with:

```bash
pip install numpy
python tensors.py
```

The full repository for this series: [github.com/grahamroy/algorithms-in-python](https://github.com/grahamroy/algorithms-in-python)
