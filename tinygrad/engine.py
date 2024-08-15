from __future__ import annotations


class Value:
    """Represents the node in computational graph."""

    def __init__(self, data: int | float, _children: set[Value] = (), _op: str = ""):
        self.data = data
        self.grad = 0
        # internal stuffs for autograd graph construction
        self._backward = lambda: None  # default behavior does nothing
        self._prev = set(_children)
        self._op = _op  # used for debugging and visualization

    def __add__(self, other: Value | int | float) -> Value:  # self + other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Value | int | float) -> Value:  # self * other
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other: int | float) -> Value:  # self ** other
        assert isinstance(other, (int, float)), "Only support int/float exp now."
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += out.grad * other * self.data ** (other - 1)

        out._backward = _backward

        return out

    def backward(self) -> None:
        # topological sort
        topo: list[Value] = []
        visited = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # backprop
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> Value:  # -self
        return self * -1

    def __radd__(self, other: Value | int | float) -> Value:  # other + self
        return self + other

    def __sub__(self, other: Value | int | float) -> Value:  # self - other
        return self + (-other)

    def __rsub__(self, other: Value | int | float) -> Value:  # self * other
        return (-self) + other

    def __rmul__(self, other: Value | int | float) -> Value:  # other * self
        return self * other

    def __truediv__(self, other: Value | int | float) -> Value:  # self / other
        return self * other**-1

    def __rtruediv__(self, other: Value | int | float) -> Value:  # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}, op={self._op})"

    def relu(val: Value | int | float) -> Value:
        val = val if isinstance(val, Value) else Value(val)

        out = Value(0 if val.data < 0 else val.data, (val,), "relu")

        def _backward():
            if out.data > 0:
                val.grad += out.grad

        out._backward = _backward

        return out
