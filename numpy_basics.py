import numpy as np

print("===== NUMPY BASICS FOR MACHINE LEARNING =====\n")

# 1. Create array using np.array()
arr1 = np.array([1, 2, 3, 4, 5])
print("1. np.array():", arr1)

# 2. Create zero array
zeros_arr = np.zeros((2, 3))
print("\n2. np.zeros():\n", zeros_arr)

# 3. Create ones array
ones_arr = np.ones((3, 2))
print("\n3. np.ones():\n", ones_arr)

# 4. Create range array
range_arr = np.arange(1, 11)
print("\n4. np.arange():", range_arr)

# 5. Reshape array
reshaped_arr = range_arr.reshape(2, 5)
print("\n5. reshape():\n", reshaped_arr)

# 6. Array slicing
slice_arr = range_arr[2:7]
print("\n6. Slicing:", slice_arr)

# 7. Element-wise addition
add_arr = arr1 + 10
print("\n7. Addition:", add_arr)

# 8. Element-wise multiplication
mul_arr = arr1 * 2
print("\n8. Multiplication:", mul_arr)

# 9. Dot product
arr2 = np.array([5, 4, 3, 2, 1])
dot_product = np.dot(arr1, arr2)
print("\n9. Dot Product:", dot_product)

# 10. Mean
mean_val = np.mean(arr1)
print("\n10. Mean:", mean_val)

# 11. Median
median_val = np.median(arr1)
print("11. Median:", median_val)

# 12. Standard Deviation
std_val = np.std(arr1)
print("12. Standard Deviation:", std_val)

# 13. Variance
var_val = np.var(arr1)
print("13. Variance:", var_val)

# 14. Broadcasting
broadcast_arr = arr1 + np.array([10])
print("\n14. Broadcasting:", broadcast_arr)

# 15. Max and Min
print("\n15. Max:", np.max(arr1))
print("    Min:", np.min(arr1))

print("\n===== END OF NUMPY OPERATIONS =====")
