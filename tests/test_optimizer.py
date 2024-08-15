import pytest
import torch
from tinygrad.engine import Value
from tinygrad.optimizer import SGD

def test_zero_grad():
    params = [Value(1.0), Value(2.0), Value(3.0)]
    optimizer = SGD(params, lr=0.1)

    # Set some gradients
    for p in params:
        p.grad = 1.0

    # Zero the gradients
    optimizer.zero_grad()

    # Check if all gradients are zero
    for p in params:
        assert p.grad == 0

def test_step():
    params = [Value(1.0), Value(2.0), Value(3.0)]
    optimizer = SGD(params, lr=0.1)

    # Set some gradients
    for p in params:
        p.grad = 1.0

    # Perform a step
    optimizer.step()

    # Check if the parameters are updated correctly
    expected_values = [0.9, 1.9, 2.9]
    for p, expected in zip(params, expected_values):
        assert pytest.approx(p.data, 0.0001) == expected

def test_sgd_against_pytorch():
    # Initialize tinygrad parameters
    tinygrad_params = [Value(1.0), Value(2.0), Value(3.0)]
    tinygrad_optimizer = SGD(tinygrad_params, lr=0.1)

    # Initialize PyTorch parameters
    torch_params = [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True), torch.tensor(3.0, requires_grad=True)]
    torch_optimizer = torch.optim.SGD(torch_params, lr=0.1)

    # Set gradients
    for p in tinygrad_params:
        p.grad = 1.0
    for p in torch_params:
        p.grad = torch.tensor(1.0)

    # Perform a step
    tinygrad_optimizer.step()
    torch_optimizer.step()

    # Compare the parameters
    for tinygrad_p, torch_p in zip(tinygrad_params, torch_params):
        assert pytest.approx(tinygrad_p.data, 0.0001) == torch_p.item()