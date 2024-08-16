import pytest
import torch
from tinygrad.engine import Value

def test_addition():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()

    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch + b_torch
    c_torch.backward()

    assert c.data == pytest.approx(c_torch.item())
    assert a.grad == pytest.approx(a_torch.grad.item())
    assert b.grad == pytest.approx(b_torch.grad.item())

def test_multiplication():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()

    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch * b_torch
    c_torch.backward()

    assert c.data == pytest.approx(c_torch.item())
    assert a.grad == pytest.approx(a_torch.grad.item())
    assert b.grad == pytest.approx(b_torch.grad.item())

def test_power():
    a = Value(2.0)
    b = 3.0
    c = a ** b
    c.backward()

    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = 3.0
    c_torch = a_torch ** b_torch
    c_torch.backward()

    assert c.data == pytest.approx(c_torch.item())
    assert a.grad == pytest.approx(a_torch.grad.item())

def test_relu():
    a = Value(-2.0)
    b = a.relu()
    b.backward()

    a_torch = torch.tensor(-2.0, requires_grad=True)
    b_torch = torch.relu(a_torch)
    b_torch.backward()

    assert b.data == pytest.approx(b_torch.item())
    assert a.grad == pytest.approx(a_torch.grad.item())

def test_combined_operations():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a ** 2 + b.relu()
    c.backward()

    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch * b_torch + a_torch ** 2 + torch.relu(b_torch)
    c_torch.backward()

    assert c.data == pytest.approx(c_torch.item())
    assert a.grad == pytest.approx(a_torch.grad.item())
    assert b.grad == pytest.approx(b_torch.grad.item())