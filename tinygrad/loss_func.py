from tinygrad.engine import Value

class MSELoss:
    def __call__(self, y: list[Value], y_hat: list[Value]) -> Value:
        loss = Value(0)
        if len(y) != len(y_hat):
            raise IndexError
        for i in range(len(y)):
            loss += (y[i] - y_hat[i]) ** 2
        return loss / Value(len(y))