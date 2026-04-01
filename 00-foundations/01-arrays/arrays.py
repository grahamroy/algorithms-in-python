"""
Arrays — Where Every Algorithm Begins
Algorithms in Python — Foundations, Part 1

Demonstrates Python lists and NumPy arrays: the four basic operations,
iteration, contiguous memory, and vectorised arithmetic.
"""

import numpy as np

# =============================================================================
# Part 1 — Python lists (array-like behaviour)
# =============================================================================

# Define array to store temperatures
temperatures = [30, 32, 28, 31, 29]

# --- Access: read a value by index (O(1)) ---
print("=== Python list: Access ===")
print(temperatures[0])  # 30
print(temperatures[2])  # 28

# --- Modify: change a value at a known position (O(1)) ---
print("\n=== Python list: Modify ===")
temperatures[1] = 33
print(temperatures)  # [30, 33, 28, 31, 29]

# --- Append: add a value to the end (amortised O(1)) ---
print("\n=== Python list: Append ===")
temperatures.append(27)
print(temperatures)  # [30, 33, 28, 31, 29, 27]

# --- Remove: delete a value (O(n) — must find then shift) ---
print("\n=== Python list: Remove ===")
temperatures.remove(28)
print(temperatures)  # [30, 33, 31, 29, 27]

# --- Iterate: walk through every element (O(n)) ---
print("\n=== Python list: Iterate ===")
for temp in temperatures:
    print(temp)

# =============================================================================
# Part 2 — From Python lists to NumPy arrays
# =============================================================================

print("\n=== Python list vs NumPy array ===")

# Python list: each element is a separate object on the heap
py_list = [1.0, 2.0, 3.0, 4.0, 5.0]

# NumPy array: one contiguous memory block, no per-element overhead
np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

print(f"Python list type: {type(py_list)}")
print(f"NumPy array type: {type(np_array)}, dtype: {np_array.dtype}")

# =============================================================================
# Part 3 — Vectorised operations (why loops disappear)
# =============================================================================

print("\n=== Vectorised operations ===")

temperatures = np.array([30, 33, 31, 29, 27])

# Element-wise arithmetic — no loop needed
fahrenheit = temperatures * 9/5 + 32
print(f"Celsius:    {temperatures}")
print(f"Fahrenheit: {fahrenheit}")

# Aggregation
print(f"\nMean: {temperatures.mean()}")
print(f"Std:  {temperatures.std():.2f}")

# Boolean indexing — filter without a loop
hot_days = temperatures[temperatures > 30]
print(f"\nDays above 30°C: {hot_days}")
