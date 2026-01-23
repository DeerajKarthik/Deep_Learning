import tensorflow as tf

# Vector
v = tf.constant([1., 2., 3.])

# Matrix
A = tf.constant([[1., 2.],
                 [3., 4.]])

B = tf.constant([[5., 6.],
                 [7., 8.]])

print(v)
print(A)

# Element-wise operations
print(A + B)
print(A - B)
print(A * B)
print(A / B)

# Scalar operations
print(A * 10)
print(A + 5)

# Matrix multiplication
C = tf.matmul(A, B)
print(C)

# Using @ operator
print(A @ B)

A_T = tf.transpose(A)
print(A_T)

det_A = tf.linalg.det(A)
print(det_A)

A_inv = tf.linalg.inv(A)
print(A_inv)

# Verify A * A⁻¹ = I
print(tf.matmul(A, A_inv))

trace_A = tf.linalg.trace(A)
print(trace_A)

eig_vals, eig_vecs = tf.linalg.eig(A)
print(eig_vals)
print(eig_vecs)


A = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

b = tf.constant([10., 20., 30.])

print(A + b)


A = tf.Variable([[1., 2.],
                 [3., 4.]])

with tf.GradientTape() as tape:
    y = tf.linalg.det(A)

grads = tape.gradient(y, A)
print(grads)


A = tf.random.uniform((2, 3, 4))
B = tf.random.uniform((2, 4, 5))

C = tf.matmul(A, B)
print(C.shape)


I = tf.eye(3)
print(I)

diag = tf.linalg.diag([1., 2., 3.])
print(diag)
