"""
Tests for Valye class
Run all with:
python -m unittest test_engine.TestValue
"""
import unittest
import torch
from torch.nn import Softplus, Sigmoid
from micrograd.engine import Value


class TestValue(unittest.TestCase):

    def test_softplus(self):
        Value.do_autograd = True
        a = Value(-4.0)
        b = Value(1.0)
        c = 0.3 * a + 0.5 * b + 1.0
        y = c.softplus()
        y.backward()
        amg, bmg, ymg = a, b, y

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([1.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = 0.3 * a + 0.5 * b + 1.0
        y = Softplus()(c)
        y.backward()
        apt, bpt, ypt = a, b, y

        tol = 1e-3

        print(f"Micrograd: a.data: {amg.data}, Torch: a.data: {apt.data.item()}")
        assert abs(amg.data - apt.data.item()) <= tol

        print(f"Micrograd: a.grad: {amg.grad}, Torch: a.grad: {apt.grad.item()}")
        assert abs(amg.grad - apt.grad.item()) <= tol

    def test_sigmoid(self):
        Value.do_autograd = True
        a = Value(-4.0)
        b = Value(1.0)
        c = 0.3 * a + 0.5 * b + 1.0
        y = c.sigmoid()
        y.backward()
        amg, bmg, ymg = a, b, y

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([1.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = 0.3 * a + 0.5 * b + 1.0
        y = Sigmoid()(c)
        y.backward()
        apt, bpt, ypt = a, b, y

        tol = 1e-3

        print(f"Micrograd: a.data: {amg.data}, Torch: a.data: {apt.data.item()}")
        assert abs(amg.data - apt.data.item()) <= tol

        print(f"Micrograd: a.grad: {amg.grad}, Torch: a.grad: {apt.grad.item()}")
        assert abs(amg.grad - apt.grad.item()) <= tol

    def test_relu(self):
        Value.do_autograd = True
        a = Value(-4.0)
        b = Value(1.0)
        c = 0.3 * a + 0.5 * b + 1.0
        y = c.relu()
        y.backward()
        amg, bmg, ymg = a, b, y

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([1.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = 0.3 * a + 0.5 * b + 1.0
        y = c.relu()
        y.backward()
        apt, bpt, ypt = a, b, y

        tol = 1e-3

        print(f"Micrograd: a.data: {amg.data}, Torch: a.data: {apt.data.item()}")
        assert abs(amg.data - apt.data.item()) <= tol

        print(f"Micrograd: a.grad: {amg.grad}, Torch: a.grad: {apt.grad.item()}")
        assert abs(amg.grad - apt.grad.item()) <= tol

    def test_relu_autograd(self):
        Value.do_autograd = True
        a = Value(-4.0)
        b = Value(1.0)
        c = 0.3 * a + 0.5 * b + 1.0
        y = c.relu()
        y.backward()
        amg, bmg, ymg = a, b, y

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([1.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = 0.3 * a + 0.5 * b + 1.0
        y = c.relu()
        y.backward()
        apt, bpt, ypt = a, b, y

        tol = 1e-3

        print(f"Micrograd: a.data: {amg.data}, Torch: a.data: {apt.data.item()}")
        assert abs(amg.data - apt.data.item()) <= tol

        print(f"Micrograd: a.grad: {amg.grad}, Torch: a.grad: {apt.grad.item()}")
        assert abs(amg.grad - apt.grad.item()) <= tol

    def test_sanity_check(self):
        Value.do_autograd = True

        # ridiculous expression to test everything
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        a = z.tanh() + z * x
        b = (z * z).tanh()
        c = z.softplus() + z * x
        d = (z * z).softplus()
        y = (h + q + q * x + a * b + c * d).sigmoid()

        y.backward()
        xmg, ymg = x, y

        # ridiculous expression to test everything
        x = torch.Tensor([-4.0]).double()
        x.requires_grad = True
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        a = z.tanh() + z * x
        b = (z * z).tanh()
        c = Softplus()(z) + z * x
        d = Softplus()(z * z)
        y = Sigmoid()(h + q + q * x + a * b + c * d)

        y.backward()
        xpt, ypt = x, y

        tolerance = 1e-3
        # forward pass went well
        print(f"ymg.data: {ymg.data}, ypt.data: {ypt.data.item()}")
        assert abs(ymg.data - ypt.data.item()) <= tolerance
        # backward pass went well
        print(f"xmg.grad: {xmg.grad}, xpt.grad: {xpt.grad.item()}")
        assert abs(xmg.grad - xpt.grad.item()) <= tolerance

    def test_more_ops(self):
        Value.do_autograd = True
        # even more ridiculous expression to test everything
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b ** 3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g += 10.0 / f
        g.backward()
        amg, bmg, gmg = a, b, g

        # even more ridiculous expression to test everything
        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b ** 3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-4
        # forward pass went well
        print(f"gmg.data: {gmg.data}, gpt.data: {gpt.data.item()}")
        assert abs(gmg.data - gpt.data.item()) < tol

        # backward pass went well
        print(f"amg.grad: {amg.grad}, apt.grad: {apt.grad.item()}")
        assert abs(amg.grad - apt.grad.item()) < tol

        print(f"bmg.grad: {bmg.grad}, bpt.grad: {bpt.grad.item()}")
        assert abs(bmg.grad - bpt.grad.item()) < tol


if __name__ == '__main__':
    unittest.main()
