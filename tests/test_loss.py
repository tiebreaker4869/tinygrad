import pytest
from tinygrad.engine import Value
from tinygrad.loss_func import MSELoss

def test_mse_loss():
    mse_loss = MSELoss()

    # Test case 1: Simple case where y and y_hat are the same
    y = [Value(1.0), Value(2.0), Value(3.0)]
    y_hat = [Value(1.0), Value(2.0), Value(3.0)]
    loss = mse_loss(y, y_hat)
    assert loss.data == 0.0

    # Test case 2: Non-zero loss
    y = [Value(1.0), Value(2.0), Value(3.0)]
    y_hat = [Value(2.0), Value(2.0), Value(4.0)]
    loss = mse_loss(y, y_hat)
    expected_loss = ((1-2)**2 + (2-2)**2 + (3-4)**2) / 3
    assert pytest.approx(loss.data, 0.0001) == expected_loss

    # Test case 3: Different lengths (should raise an error)
    y = [Value(1.0), Value(2.0)]
    y_hat = [Value(1.0), Value(2.0), Value(3.0)]
    with pytest.raises(IndexError):
        mse_loss(y, y_hat)