package neural

import (
  // "fmt"
  "math/big"
  "crypto/rand"
)

// Set of weights + bias linked to a layer
type Neuron struct {
  // Amount of weights
  MaxInputs int `json:"-"`
  // Set of weights
  Weights []float64 `json:"Weights"` // []*float64?
  // Bias
  Bias float64 `json:"Bias"`
  // Previous momentum of every weight and bias
  Momentums []float64 `json:"-"`
  // Layer to which neuron is linked
  Layer *Layer `json:"-"`
  activation float64
  delta float64
  error float64
  Inputs []float64 `json:"-"`
}

// Creates a neuron linked to a layer
func NewNeuron (Layer *Layer, MaxInputs int) *Neuron {
  neuron := &Neuron{
    MaxInputs: MaxInputs,
    Weights: make([]float64, MaxInputs),
    Bias: randomFloat(-1.0, 1.0),
    Momentums: make([]float64, MaxInputs + 1),
    Layer: Layer,
    Inputs: make([]float64, MaxInputs),
  }

  for i := 0; i < neuron.MaxInputs; i++ {
    neuron.Weights[i] = randomFloat(-1.0, 1.0)
  }

  return neuron
}

// Process neuron forward based on inputs
func (neuron *Neuron) Think (inputs []float64) float64 {
  sum := neuron.Bias;

  for i := 0; i < neuron.MaxInputs; i++ {
    sum += inputs[i] * neuron.Weights[i]
    neuron.Inputs[i] = inputs[i]
  }

  neuron.activation = neuron.Layer.Forward(sum)
  return neuron.activation
}

// Learning optimizer by momentum
func (neuron *Neuron) Optimizer (index int, value float64) float64 {
  neuron.Momentums[index] = value + (neuron.Layer.Momentum * neuron.Momentums[index])
  return neuron.Momentums[index]
}

// Clone neuron with same weights, bias, etc
func (neuron *Neuron) Clone () *Neuron {
  clone := NewNeuron(neuron.Layer, neuron.MaxInputs)

  for i := 0; i < neuron.MaxInputs; i++ {
    clone.Weights[i] = neuron.Weights[i]
  }
  clone.Bias = neuron.Bias

  return clone
}

// Mutate randomizing weights/bias based on probability
func (neuron *Neuron) Mutate (probability float64) {
  for i := 0; i < neuron.MaxInputs; i++ {
    if probability >= cryptoRandomFloat() {
      neuron.Weights[i] += randomFloat(-1.0, 1.0)
      neuron.Momentums[i] = 0.0
    }
  }

  if probability >= cryptoRandomFloat() {
    neuron.Bias += randomFloat(-1.0, 1.0)
    neuron.Momentums[neuron.MaxInputs] = 0.0
  }
}

// Crossover two neurons merging weights and bias
func (neuronA *Neuron) Crossover (neuronB Neuron, dominant float64) *Neuron {
  new := NewNeuron(neuronA.Layer, neuronA.MaxInputs)

  for i := 0; i < new.MaxInputs; i++ {
    if cryptoRandomFloat() >= 0.5 {
      new.Weights[i] = neuronA.Weights[i]
    } else {
      new.Weights[i] = neuronB.Weights[i]
    }
  }

  if cryptoRandomFloat() >= 0.5 {
    new.Bias = neuronA.Bias
  } else {
    new.Bias = neuronB.Bias
  }

  return new
}

// Reset weights, bias and momentums
func (neuron *Neuron) Reset () {
  for i := 0; i < neuron.MaxInputs; i++ {
    neuron.Weights[i] = randomFloat(-1.0, 1.0)
    neuron.Momentums[i] = 0.0
  }

  neuron.Bias = randomFloat(-1.0, 1.0)
  neuron.Momentums[neuron.MaxInputs] = 0.0
}

func randomFloat (min float64, max float64) float64 {
  return min + cryptoRandomFloat() * (max - min)
}

func cryptoRandomFloat () float64 {
  num, err := rand.Int(rand.Reader, big.NewInt(1e17))
  if err != nil {
    panic(err)
  }
  return float64(num.Int64()) / float64(1e17)
}

func randomInt (max int64) int {
  num, err := rand.Int(rand.Reader, big.NewInt(max))
  if err != nil {
    panic(err)
  }
  return int(num.Int64())
}
