import torch
from tinygrad.engine import Value

def test_value_operations():
    # Basic tests
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0

    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0

    a = Value(2.0)
    b = a ** 3
    assert b.data == 8.0

    a = Value(-1.0)
    b = Value.relu(a)
    assert b.data == 0.0
    a = Value(1.0)
    b = Value.relu(a)
    assert b.data == 1.0

    a = Value(2.0)
    b = -a
    assert b.data == -2.0

    a = Value(5.0)
    b = Value(3.0)
    c = a - b
    assert c.data == 2.0

    a = Value(6.0)
    b = Value(3.0)
    c = a / b
    assert c.data == 2.0

    a = Value(2.0)
    b = 3 + a
    assert b.data == 5.0

    a = Value(2.0)
    b = 3 - a
    assert b.data == 1.0

    a = Value(2.0)
    b = 3 * a
    assert b.data == 6.0

    a = Value(2.0)
    b = 6 / a
    assert b.data == 3.0

def test_value_backward():
    # Backward tests
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0

    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0

def test_value_with_torch():
    # Torch comparison tests
    a = torch.tensor(2.0, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3.0, requires_grad=True, dtype=torch.float32)
    c = a + b
    c.backward()
    a_val = Value(2.0)
    b_val = Value(3.0)
    c_val = a_val + b_val
    c_val.backward()
    assert torch.isclose(torch.tensor(c_val.data, dtype=torch.float32), c)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)
    assert torch.isclose(torch.tensor(b_val.grad, dtype=torch.float32), b.grad)

    a = torch.tensor(2.0, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3.0, requires_grad=True, dtype=torch.float32)
    c = a * b
    c.backward()
    a_val = Value(2.0)
    b_val = Value(3.0)
    c_val = a_val * b_val
    c_val.backward()
    assert torch.isclose(torch.tensor(c_val.data, dtype=torch.float32), c)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)
    assert torch.isclose(torch.tensor(b_val.grad, dtype=torch.float32), b.grad)

    a = torch.tensor(2.0, requires_grad=True, dtype=torch.float32)
    b = a ** 3
    b.backward()
    a_val = Value(2.0)
    b_val = a_val ** 3
    b_val.backward()
    assert torch.isclose(torch.tensor(b_val.data, dtype=torch.float32), b)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)

    a = torch.tensor(-1.0, requires_grad=True, dtype=torch.float32)
    b = torch.relu(a)
    b.backward()
    a_val = Value(-1.0)
    b_val = Value.relu(a_val)
    b_val.backward()
    assert torch.isclose(torch.tensor(b_val.data, dtype=torch.float32), b)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)

    a = torch.tensor(2.0, requires_grad=True, dtype=torch.float32)
    b = -a
    b.backward()
    a_val = Value(2.0)
    b_val = -a_val
    b_val.backward()
    assert torch.isclose(torch.tensor(b_val.data, dtype=torch.float32), b)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)

    a = torch.tensor(5.0, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3.0, requires_grad=True, dtype=torch.float32)
    c = a - b
    c.backward()
    a_val = Value(5.0)
    b_val = Value(3.0)
    c_val = a_val - b_val
    c_val.backward()
    assert torch.isclose(torch.tensor(c_val.data, dtype=torch.float32), c)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)
    assert torch.isclose(torch.tensor(b_val.grad, dtype=torch.float32), b.grad)

    a = torch.tensor(6.0, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3.0, requires_grad=True, dtype=torch.float32)
    c = a / b
    c.backward()
    a_val = Value(6.0)
    b_val = Value(3.0)
    c_val = a_val / b_val
    c_val.backward()
    assert torch.isclose(torch.tensor(c_val.data, dtype=torch.float32), c)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)
    assert torch.isclose(torch.tensor(b_val.grad, dtype=torch.float32), b.grad)

def test_value_with_torch_complex():
    # Complex operation 1: (a + b) * c
    a = torch.tensor(2.0, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3.0, requires_grad=True, dtype=torch.float32)
    c = torch.tensor(4.0, requires_grad=True, dtype=torch.float32)
    d = (a + b) * c
    d.backward()
    a_val = Value(2.0)
    b_val = Value(3.0)
    c_val = Value(4.0)
    d_val = (a_val + b_val) * c_val
    d_val.backward()
    assert torch.isclose(torch.tensor(d_val.data, dtype=torch.float32), d)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)
    assert torch.isclose(torch.tensor(b_val.grad, dtype=torch.float32), b.grad)
    assert torch.isclose(torch.tensor(c_val.grad, dtype=torch.float32), c.grad)

    # Complex operation 2: (a * b) + (c / d)
    a = torch.tensor(2.0, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3.0, requires_grad=True, dtype=torch.float32)
    c = torch.tensor(4.0, requires_grad=True, dtype=torch.float32)
    d = torch.tensor(5.0, requires_grad=True, dtype=torch.float32)
    e = (a * b) + (c / d)
    e.backward()
    a_val = Value(2.0)
    b_val = Value(3.0)
    c_val = Value(4.0)
    d_val = Value(5.0)
    e_val = (a_val * b_val) + (c_val / d_val)
    e_val.backward()
    assert torch.isclose(torch.tensor(e_val.data, dtype=torch.float32), e)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)
    assert torch.isclose(torch.tensor(b_val.grad, dtype=torch.float32), b.grad)
    assert torch.isclose(torch.tensor(c_val.grad, dtype=torch.float32), c.grad)
    assert torch.isclose(torch.tensor(d_val.grad, dtype=torch.float32), d.grad)

    # Complex operation 3: relu(a * b + c) - d
    a = torch.tensor(2.0, requires_grad=True, dtype=torch.float32)
    b = torch.tensor(3.0, requires_grad=True, dtype=torch.float32)
    c = torch.tensor(4.0, requires_grad=True, dtype=torch.float32)
    d = torch.tensor(5.0, requires_grad=True, dtype=torch.float32)
    e = torch.relu(a * b + c) - d
    e.backward()
    a_val = Value(2.0)
    b_val = Value(3.0)
    c_val = Value(4.0)
    d_val = Value(5.0)
    e_val = Value.relu(a_val * b_val + c_val) - d_val
    e_val.backward()
    assert torch.isclose(torch.tensor(e_val.data, dtype=torch.float32), e)
    assert torch.isclose(torch.tensor(a_val.grad, dtype=torch.float32), a.grad)
    assert torch.isclose(torch.tensor(b_val.grad, dtype=torch.float32), b.grad)
    assert torch.isclose(torch.tensor(c_val.grad, dtype=torch.float32), c.grad)
    assert torch.isclose(torch.tensor(d_val.grad, dtype=torch.float32), d.grad)