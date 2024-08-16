import random
from tinygrad.engine import Value

class Module:

    def __init__(self):
        super().__setattr__("_parameters", {})
        super().__setattr__("_modules", {})

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self, recurse=True):
        params = list(self._parameters.values())
        if recurse:
            for _, module in self._modules.items():
                params += module.parameters(recurse)
        return params
    
    def __setattr__(self, name, value):
        """Discover Value/Module/list[Value | Module] attribute automatically and register them to autograd parameters"""
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Value):
            self._parameters[name] = value
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, Module):
                    self._modules[f"{name}.{i}"] = v
                elif isinstance(v, Value):
                    self._parameters[f"{name}.{i}"] = v
        
        super().__setattr__(name, value)
        

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        super().__init__()
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        super().__init__()
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        super().__init__()
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"