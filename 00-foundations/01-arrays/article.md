# Arrays — Where Every Algorithm Begins

### *Algorithms in Python — Foundations, Part 1*

---

Before there are neural networks, before there are decision trees, before there is any
machine learning at all, there is a list of numbers. A row of sensor readings. A column of
prices. A sequence of pixels. Strip away every layer of abstraction in modern AI and you
will find, at the very bottom, an array.

An array is the simplest data structure in programming: an ordered collection of items
stored in contiguous memory. It is also the most important one. Every matrix is built from
arrays. Every tensor is built from matrices. Every dataset that flows through a machine
learning pipeline begins its life as values arranged in sequence. If you understand arrays
deeply — how they are stored, how they are accessed, why some operations are fast and
others are slow — you understand the foundation that everything else stands on.

This article is the first in a series on data structures for AI. We start here because
arrays are where the complexity begins — and where the intuition needs to be sharpest.

---

## What is an array?

An array is an ordered, fixed-position collection of elements. Each element sits at a
numbered index, starting from zero. Given the index, you can retrieve any element
instantly — this is called **constant-time access**, or O(1).

In Python, the built-in `list` gives you array-like behaviour out of the box:

```python
temperatures = [30, 32, 28, 31, 29]
```

Five numbers. Five positions. Position 0 holds `30`, position 4 holds `29`. That is the
entire idea. What makes it powerful is what you can *do* with it.

---

## The four basic operations

Every data structure is defined by its operations — how you read, write, grow, and shrink
it. Arrays support four fundamental ones.

### 1. Access — read a value by index

```python
print(temperatures[0])  # 30
print(temperatures[2])  # 28
```

This is the operation arrays are built for. Because elements sit in contiguous memory,
the computer can jump directly to any position without scanning through the ones before
it. Access is O(1) — the same speed whether your array has five elements or five million.

This property matters enormously in machine learning. When a neural network looks up a
word embedding, it is performing an array access. When a decision tree checks a feature
value, it is performing an array access. The O(1) guarantee is the reason these
operations are fast enough to run at scale.

### 2. Modify — change a value at a known position

```python
temperatures[1] = 33
print(temperatures)  # [30, 33, 28, 31, 29]
```

Like access, modification by index is O(1). You know where the element lives, so you
overwrite it directly. In training loops, weight updates work exactly this way — the
gradient tells you *which* parameter to change and *by how much*, and the array makes
that change instant.

### 3. Append — add a value to the end

```python
temperatures.append(27)
print(temperatures)  # [30, 33, 28, 31, 29, 27]
```

Appending is *amortised* O(1) in Python. Most of the time, there is spare capacity at
the end of the underlying memory block and the new value slots right in. Occasionally
the list runs out of room, allocates a bigger block, and copies everything across — but
Python over-allocates deliberately, so this happens rarely enough that the average cost
stays constant.

This is the operation you use when building a dataset incrementally — reading rows from a
file, collecting predictions during inference, or logging metrics during training.

### 4. Remove — delete a value

```python
temperatures.remove(28)
print(temperatures)  # [30, 33, 31, 29, 27]
```

Removal is where arrays show their weakness. `remove(28)` has to *find* the value first
(O(n) scan), then shift every element after it one position to the left (another O(n)
operation). For a five-element list this is invisible. For a million-element list, it
starts to hurt.

This is why data pipelines that need frequent insertions and deletions in the middle of a
sequence often reach for a different structure — a linked list, a set, or a deque. Arrays
are optimised for reading and writing at known positions, not for reshuffling.

---

## Iterating through an array

The most common thing you will do with an array is walk through it, element by element:

```python
for temp in temperatures:
    print(temp)
```

This is O(n) — you visit every element exactly once. Iteration is the backbone of
virtually every algorithm. Computing a mean, finding a maximum, calculating a loss
function, applying a transformation to every data point — all of these are loops over
arrays.

In machine learning, the training loop itself is an iteration over an array of batches.
Each batch is an array of samples. Each sample is an array of features. Arrays all the
way down.

---

## From lists to NumPy arrays

Python lists are flexible — they can hold mixed types, resize freely, and nest
arbitrarily. But that flexibility has a cost. Each element is a full Python object stored
at a separate location in memory, with pointers connecting them. For numerical work, this
overhead is devastating.

NumPy replaces Python lists with **ndarray** — a contiguous block of memory holding
values of a single type (typically `float64`). The difference is not subtle:

```python
import numpy as np

# Python list: each element is a separate object on the heap
py_list = [1.0, 2.0, 3.0, 4.0, 5.0]

# NumPy array: one contiguous memory block, no per-element overhead
np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
```

The NumPy array stores its five `float64` values in 40 consecutive bytes. The Python list
stores five pointers to five separate float objects scattered across memory. When you
multiply every element by two, NumPy does it in a single C-level loop over that
contiguous block. Python does it with five separate object lookups, five separate method
dispatches, and five separate memory allocations for the results.

This is why NumPy exists. This is why every machine learning library — scikit-learn,
PyTorch, TensorFlow, JAX — is built on top of arrays stored in contiguous memory. The
data structure did not change. The memory layout did.

---

## Vectorised operations — why loops disappear

Once your data lives in a NumPy array, you stop writing loops and start writing
*expressions*:

```python
import numpy as np

temperatures = np.array([30, 33, 31, 29, 27])

# Element-wise arithmetic — no loop needed
fahrenheit = temperatures * 9/5 + 32
print(fahrenheit)  # [86.  91.4 87.8 84.2 80.6]

# Aggregation
print(temperatures.mean())  # 30.0
print(temperatures.std())   # 2.0

# Boolean indexing — filter without a loop
hot_days = temperatures[temperatures > 30]
print(hot_days)  # [33 31]
```

Each of these operations runs in compiled C under the hood, operating on the raw memory
block without touching the Python interpreter. This is called **vectorisation**, and it is
the single most important performance technique in numerical Python.

When you see a machine learning formula like:

> ŷ = Xw + b

that is not pseudocode. That is a real NumPy expression — a matrix-vector multiplication
followed by a broadcast addition — and it runs at near-C speed because every piece of it
is a vectorised operation on contiguous arrays.

---

## Arrays in the machine learning pipeline

Arrays are not just the first data structure you learn. They are the data structure you
use at every stage of the pipeline:

**Data loading.** A CSV file becomes a 2D array: rows are samples, columns are features.
This is the format scikit-learn expects, and it is the format pandas DataFrames store
internally.

**Feature engineering.** Scaling, normalising, one-hot encoding — all of these are
vectorised operations on arrays. You rarely write a loop.

**Model parameters.** A linear regression's weights are a 1D array. A neural network's
weights are a collection of 2D arrays (matrices). Every parameter update during training
is an array operation.

**Predictions.** The output of `model.predict(X)` is an array. The loss function compares
two arrays — predictions and targets — element by element.

**Evaluation.** Accuracy, precision, recall, MSE — all computed by comparing arrays.

The array is the universal interface of machine learning. Master it, and every library,
every framework, and every algorithm becomes easier to understand.

---

## What comes next

Arrays store data in a single dimension — a flat sequence of values. But real-world data
almost always has more structure. A spreadsheet has rows and columns. An image has height,
width, and colour channels. A batch of images adds a fourth dimension.

In the next article, we will take the array and extend it into two dimensions: the
**matrix**. Matrices are where linear algebra enters the picture, and linear algebra is
the language machine learning thinks in. Every operation we covered here — access,
modification, vectorised arithmetic — generalises naturally into matrices and beyond.

The journey from a simple list of temperatures to a tensor flowing through a neural
network is shorter than it looks. It starts here.

---

## The complete code

The full script is on GitHub — grab it here and run it yourself:

[**arrays.py**](https://github.com/grahamroy/algorithms-in-python/blob/main/00-foundations/01-arrays/arrays.py)

Run it with:

```bash
python arrays.py
```

The full repository for this series: [github.com/grahamroy/algorithms-in-python](https://github.com/grahamroy/algorithms-in-python)
