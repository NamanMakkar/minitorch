import minitorch

def f(x, y):
    return (x * 3) * y + 10 * x

x, y = minitorch.Scalar(3), minitorch.Scalar(4)
print(f(x, y).backward(2))
print(x.derivative)
