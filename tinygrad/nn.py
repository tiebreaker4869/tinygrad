from abc import ABC, abstractmethod
import random
from tinygrad.engine import Value
from typing import Any, List

class Module(ABC):

    def __init__(self):
        self._parameters = []
        self._modules = {}

    def parameters(self) -> List[Value]:
        params = self._parameters[:]
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    @abstractmethod
    def forward(self, x: List[int | float | Value]) -> List[Value]:
        pass

    def __call__(self, x: List[int | float | Value]) -> List[Value]:
        return self.forward(x)
    
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_parameters", "_modules"}:
            object.__setattr__(self, name, value)
        else:
            if isinstance(value, Module):
                self._modules[name] = value
            elif isinstance(value, Value):
                if value not in self._parameters:
                    self._parameters.append(value)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, Module):
                        self._modules[name] = v
                    elif isinstance(v, Value):
                        if v not in self._parameters:
                            self._parameters.append(v)
            object.__setattr__(self, name, value)

class Neuron(Module):
    def __init__(self, in_features: int, nonlin=True):
        super().__init__()
        self.w = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
        self.b = Value(0)
        self.nonlin = nonlin
    
    def forward(self, x: List[int | float | Value]) -> List[Value]:
        activation = [sum([x[i] * self.w[i] for i in range(len(x))]) + self.b]
        if self.nonlin:
            activation = [Value.relu(act) for act in activation]
        return activation

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, nonlin=True):
        super().__init__()
        self.neurons = [Neuron(in_features, nonlin) for _ in range(out_features)]

    def forward(self, x: List[int | float | Value]) -> List[Value]:
        return [neuron(x)[0] for neuron in self.neurons]    


class MLP(Module):
    def __init__(self, in_features: int, hidden_features: List[int], out_features: int):
        super().__init__()
        self.hidden = [Linear(in_features, hidden_features[0])]
        self.hiddens = [Linear(hidden_features[i], hidden_features[i+1]) for i in range(len(hidden_features)-1)]
        self.out = Linear(hidden_features[-1], out_features)

    def forward(self, x: List[int | float | Value]) -> List[Value]:
        for layer in self.hidden + self.hiddens:
            x = layer(x)
        return self.out(x)