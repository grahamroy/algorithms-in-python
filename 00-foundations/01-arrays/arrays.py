# Define array to store temperatures
temperatures = [30, 32, 28, 31, 29]

# Accessing elements in the array
print(temperatures[0])  # Output: 30
print(temperatures[2])  # Output: 28

# Modifying elements in the array
temperatures[1] = 33
print(temperatures)  # Output: [30, 33, 28, 31, 29]

# Adding new elements to the array
temperatures.append(27)
print(temperatures)  # Output: [30, 33, 28, 31, 29, 27]

# Removing elements from the array
temperatures.remove(28)
print(temperatures)  # Output: [30, 33, 31, 29, 27]
# Iterating through the array
for temp in temperatures:
    print(temp)     
# Output:
# 30
# 33
# 31
# 29
# 27


