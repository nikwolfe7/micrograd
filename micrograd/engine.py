"""Library Containing Basic Classes for Value & Backpropagation."""

import math


class Value:
    """
    Class for creating scalar value object & storing its gradient.
    """
    # used for computing gradients
    h = 0.00000001
    # TODO(nikwolfe7): Remove autograd flag when we confirm it works
    do_autograd = True

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def _autograd(self, func, args, wrt):
        # find which arg we're deriving with respect to
        h_args = []
        for arg in args:
            # must point to same reference
            if wrt is arg:
                # arg must be a new value object
                arg = Value(arg.data)
                arg.data += self.h
            h_args.append(arg)
        h_args = tuple(h_args)
        # definition of derivative
        # (f(a+h) - f(a)) / h
        g = (func(*h_args) - func(*args)) * (self.h ** -1)
        return g

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=str(other))

        def _add(a, b):
            return a.data + b.data

        v = _add(self, other)
        out = Value(v, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=str(other))

        def _mul(a, b):
            return a.data * b.data

        v = _mul(self, other)
        out = Value(v, (self, other), '*')

        def _backward():
            if self.do_autograd:
                self.grad += self._autograd(_mul, (self, other), wrt=self) * out.grad
                other.grad += self._autograd(_mul, (self, other), wrt=other) * out.grad
            else:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        def _tanh(a):
            return (math.exp(2 * a.data) - 1.0) / (math.exp(2 * a.data) + 1.0)

        v = _tanh(self)
        out = Value(v, _children=(self,), _op='tanh')

        def _backward():
            if self.do_autograd:
                self.grad += self._autograd(_tanh, (self,), wrt=self) * out.grad
            else:
                # d/dx tanh(x) = 1 - tanh(x)^2
                self.grad += (1.0 - v ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        def _sigmoid(a):
            # 1 / (1 + e^(-x))
            return (1 + math.exp(-a.data)) ** -1

        v = _sigmoid(self)
        out = Value(v, _children=(self,), _op='sigmoid')

        def _backward():
            if self.do_autograd:
                self.grad += self._autograd(_sigmoid, (self,), wrt=self) * out.grad
            else:
                # d/dx sig(x) = sig(x)(1 - sig(x))
                self.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward
        return out

    def softplus(self):
        def _softplus(a):
            # f(x) = log(1 + exp(x))
            return math.log(1 + math.exp(a.data))

        v = _softplus(self)
        out = Value(v, _children=(self,), _op='softplus')

        def _backward():
            if self.do_autograd:
                self.grad += self._autograd(_softplus, (self,), wrt=self) * out.grad
            else:
                # d/dx softplus(x) = sigmoid(x)
                self.grad += self.sigmoid().data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        def _relu(a):
            return max(0, a.data)

        v = _relu(self)
        out = Value(v, _children=(self,), _op='ReLU')

        def _backward():
            if self.do_autograd:
                self.grad += self._autograd(_relu, (self,), wrt=self) * out.grad
            else:
                self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        def _pow(a, b):
            return a.data ** b

        v = _pow(self, other)
        out = Value(v, (self,), f"**{other}")

        def _backward():
            if self.do_autograd:
                self.grad += self._autograd(_pow, (self, other), self) * out.grad
            else:
                # power rule...
                self.grad += other * _pow(self, other - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """returns e^(data)"""

        def _exp(a):
            return math.exp(a.data)

        v = _exp(self)
        out = Value(v, (self,), _op='exp')

        def _backward():
            if self.do_autograd:
                self.grad += self._autograd(_exp, (self,), wrt=self) * out.grad
            else:
                # d/dx e^x = e^x
                self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # subtraction
        return self + (-other)

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rmul__(self, other):  # other is self
        return self * other

    def __radd__(self, other):  # other is self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # run backprop in reverse topological order
        self.grad = 1.0
        for val in reversed(topo):
            val._backward()

