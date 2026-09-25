class Tensor:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += result.grad
            other.grad += result.grad

        result._backward = _backward

        return result

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * result.grad
            other.grad += self.data * result.grad

        result._backward = _backward

        return result

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        result = Tensor(
            self.data / other.data,
            (self, other),
            "/"
        )

        def _backward():
            self.grad += (1.0 / other.data) * result.grad
            other.grad += (
                -self.data /
                (other.data ** 2)
            ) * result.grad

        result._backward = _backward

        return result

    def exp(self):
        import math

        result = Tensor(
            math.exp(self.data),
            (self,),
            "exp"
        )

        def _backward():
            self.grad += result.data * result.grad

        result._backward = _backward

        return result

    def log(self):
        import math

        result = Tensor(
            math.log(self.data),
            (self,),
            "log"
        )

        def _backward():
            self.grad += (1.0 / self.data) * result.grad

        result._backward = _backward
        return result

    def __pow__(self, power):
        result = Tensor(
            self.data ** power,
            (self,),
            f"**{power}"
        )

        def _backward():
            self.grad += (
                power *
                (self.data ** (power - 1)) *
                result.grad
            )

        result._backward = _backward

        return result

    def backward(self):
        topology = []
        visited = set()

        def build_topology(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topology(child)

                topology.append(node)

        build_topology(self)

        self.grad = 1.0

        for node in reversed(topology):
            node._backward()


if __name__ == "__main__":
    x = Tensor(2.0)
    target = Tensor(5.0)

    error = x - target
    loss = error ** 2

    loss.backward()

    print("x:", x)
    print("target:", target)
    print("error:", error)
    print("loss:", loss)
