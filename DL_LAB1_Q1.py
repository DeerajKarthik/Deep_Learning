import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([[1, 2], [3, 4]])

print(a)
print(b)

c = torch.tensor([1, 2, 3], dtype=torch.float32)
d = torch.tensor([1, 2, 3], dtype=torch.int64)

print(c.dtype)
print(d.dtype)

# Common initialization methods
zeros = torch.zeros((2, 3))
ones = torch.ones((2, 3))
rand = torch.rand((2, 3))        # Uniform [0, 1)
randn = torch.randn((2, 3))      # Normal distribution

print(zeros)
print(ones)
print(rand)
print(randn)

x = torch.tensor([[5., 6.], [7., 8.]])
zeros_like_x = torch.zeros_like(x)
ones_like_x = torch.ones_like(x)

print(zeros_like_x)
print(ones_like_x)

x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])

print(x + y)
print(x - y)
print(x * y)
print(x / y)

print(x + 10)
print(x * 2)

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([10, 20, 30])

print(a + b)   # b is broadcast across rows

t = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print(t[0])        # First row
print(t[:, 1])    # Second column
print(t[1, 2])    # Element at (1,2)

# Boolean indexing
mask = t > 3
print(mask)
print(t[mask])

x = torch.arange(12)

print(x)
print(x.shape)

# Reshape
y = x.reshape(3, 4)
print(y)

# Reshape
y = x.reshape(3, 4)
print(y)

# Flatten
flat = y.flatten()
print(flat)

x = torch.tensor([2.0, 3.0], requires_grad=True)

y = x[0]**2 + 3 * x[1]
print(y)

y.backward()
print(x.grad)

A = torch.tensor([[1., 2.],
                  [3., 4.]], requires_grad=True)

B = torch.tensor([[2., 0.],
                  [1., 2.]])

C = torch.sum(A @ B)
print(C)


C.backward()
print(A.grad)

with torch.no_grad():
    z = A * 10

print(z.requires_grad)

