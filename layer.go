package neural

// Layer is a set of neurons + config
type Layer struct {
  // Amount of inputs (default is previous layer units)
  Inputs int `json:"-"`
  // Amount of neurons
  Units int `json:"-"`
  // Slice of neurons
  Neurons []*Neuron `json:"Neurons"`
  // Default activation is sigmoid
  Activation string `json:"Activation,omitempty"`
  // Activation function
  Forward ForwardFn `json:"-"`
  // Derivative activation function
  Backward BackwardFn `json:"-"`
  // Default loss is mse
  Loss string `json:"Loss,omitempty"`
  // Loss function
  LossFn LossFn `json:"-"`
  // Default rate is 0.001
  Rate float64 `json:"-"`
  // Default momentum is 0.999
  Momentum float64 `json:"-"`
  // Range of arbitrary values for input/output layers
  Range [][]float64 `json:"Range,omitempty"`
}

// NewLayer creates a layer based on simple layer definition
func NewLayer (layer *Layer) *Layer {
  if layer.Rate == 0.0 {
    layer.Rate = 0.001
  }

  if layer.Momentum == 0.0 {
    layer.Momentum = 0.999
  }
  
  layer.Neurons = make([]*Neuron, layer.Units)
  for i := 0; i < layer.Units; i++ {
    layer.Neurons[i] = NewNeuron(layer, layer.Inputs)
  }

  activation := layer.SetActivation(layer.Activation)

  if len(activation.Ranges) == 0 {
    layer.Range = [][]float64{}
  }

  for i, total := 0, len(layer.Range); i < total; i++ {
    layer.Range[i] = append(layer.Range[i], activation.Ranges[0], activation.Ranges[1])
  }

  return layer
}

// Think process the layer forward based on inputs
func (layer *Layer) Think (inputs []float64) []float64 {
  outs := make([]float64, layer.Units)

  for i := 0; i < layer.Units; i++ {
    outs[i] = layer.Neurons[i].Think(inputs)
  }

  return outs
}

// Clone layer with same neurons, activation, range, etc
func (layer *Layer) Clone () *Layer {
  clone := NewLayer(&Layer{
    Inputs: layer.Inputs,
    Units: layer.Units,
    Activation: layer.Activation,
    Rate: layer.Rate,
    Momentum: layer.Momentum,
  })

  for i := 0; i < clone.Units; i++ {
    clone.Neurons[i] = layer.Neurons[i].Clone()
  }

  clone.Range = make([][]float64, len(layer.Range))
  copy(clone.Range, layer.Range)

  return clone
}

// Mutate neurons of layer based on probability
func (layer *Layer) Mutate (probability float64) {
  for i := 0; i < layer.Units; i++ {
    layer.Neurons[i].Mutate(probability)
  }
}

// Crossover two layers merging neurons
func (layer *Layer) Crossover (layerB *Layer, dominant float64) *Layer {
  new := NewLayer(&Layer{
    Inputs: layer.Inputs,
    Units: layer.Units,
    Activation: layer.Activation,
    Rate: layer.Rate,
    Momentum: layer.Momentum,
  })

  for i := 0; i < layer.Units; i++ {
    new.Neurons[i] = layer.Neurons[i].Crossover(*layerB.Neurons[i], dominant)
  }

  new.Range = make([][]float64, len(layer.Range))
  copy(new.Range, layer.Range)

  return new
}

// Reset every neuron (weights, bias, etc)
func (layer *Layer) Reset () {
  for i := 0; i < layer.Units; i++ {
    layer.Neurons[i].Reset()
  }
}

// SetActivation set or change the activation functions based on name
func (layer *Layer) SetActivation (activation string) ActivationSet {
  set := selectActivation(activation)
  layer.Forward = set.Forward
  layer.Backward = set.Backward
  return set
}