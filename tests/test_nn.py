from tinygrad.engine import Value
from tinygrad.nn import Module, Neuron, Linear, MLP


def test_module_parameters():
    class TestModule(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    module = TestModule()
    assert module.parameters() == []


def test_neuron_forward():
    in_features = 3
    x = [Value(1.0), Value(2.0), Value(3.0)]
    neuron = Neuron(in_features)
    output = neuron.forward(x)
    assert len(output) == 1
    assert isinstance(output[0], Value)


def test_linear_forward():
    in_features = 3
    out_features = 2
    x = [Value(1.0), Value(2.0), Value(3.0)]
    linear = Linear(in_features, out_features)
    output = linear.forward(x)
    assert len(output) == out_features
    assert all(isinstance(o, Value) for o in output)


def test_mlp_forward():
    in_features = 3
    hidden_features = [4, 5]
    out_features = 2
    x = [Value(1.0), Value(2.0), Value(3.0)]
    mlp = MLP(in_features, hidden_features, out_features)
    output = mlp.forward(x)
    assert len(output) == out_features
    assert all(isinstance(o, Value) for o in output)


def test_module_call():
    class TestModule(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    module = TestModule()
    x = [Value(1.0)]
    output = module(x)
    assert output == x


def test_module_setattr():
    class TestModule(Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    module = TestModule()
    value = Value(1.0)
    module.value = value
    assert module.parameters() == [value]

    submodule = TestModule()

    module.submodule = submodule

    submodule.value = Value(2.0)

    assert module.parameters() == [value, submodule.value]
