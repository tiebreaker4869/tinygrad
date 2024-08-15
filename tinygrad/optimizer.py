from abc import ABC, abstractmethod

from tinygrad.engine import Value

class Optimizer(ABC):    
    @abstractmethod    
    def zero_grad():
        pass

    @abstractmethod
    def step():
        pass

class SGD(Optimizer):
    def __init__(self, params: list[Value], lr: float=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad