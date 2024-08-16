from tinygrad.engine import Value
from tinygrad.nn import Neuron, Layer, MLP

def test_neuron_initialization():
    nin = 3
    neuron = Neuron(nin)
    assert len(neuron.w) == nin
    assert isinstance(neuron.b, Value)
    assert neuron.nonlin is True

def test_neuron_call():
    nin = 3
    neuron = Neuron(nin)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    result = neuron(x)
    assert isinstance(result, Value)

def test_layer_initialization():
    nin = 3
    nout = 2
    layer = Layer(nin, nout)
    assert len(layer.neurons) == nout
    for neuron in layer.neurons:
        assert isinstance(neuron, Neuron)

def test_layer_call():
    nin = 3
    nout = 2
    layer = Layer(nin, nout)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    result = layer(x)
    assert isinstance(result, list)
    assert len(result) == nout
    for res in result:
        assert isinstance(res, Value)

def test_mlp_initialization():
    nin = 3
    nouts = [4, 5, 6]
    mlp = MLP(nin, nouts)
    assert len(mlp.layers) == len(nouts)
    for layer in mlp.layers:
        assert isinstance(layer, Layer)

def test_mlp_call():
    nin = 3
    nouts = [4, 5, 6]
    mlp = MLP(nin, nouts)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    result = mlp(x)
    assert isinstance(result, list)
    assert len(result) == nouts[-1]
    for res in result:
        assert isinstance(res, Value)