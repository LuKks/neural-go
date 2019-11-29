package neural

import (
  "time"
  "math"
  "math/rand"
)

type Neuron struct {
  maxInputs int
  inputs []float64
  weights []float64
  bias float64
  activation float64
  delta float64
  error float64
}

type Layer struct {
  maxNeurons int
  neurons []Neuron
  inputMin []float64
  inputMax []float64
}
var layerPrevNeurons int

type Neural struct {
  maxLayers int
  layers []Layer
}

func NewNeuron (maxInputs int) Neuron {
  neuron := Neuron{
    maxInputs: maxInputs,
    inputs: make([]float64, maxInputs),
    weights: make([]float64, maxInputs),
    bias: randomFloat(-1.0, 1.0),
  }

  for i := 0; i < neuron.maxInputs; i++ {
    neuron.weights[i] = randomFloat(-1.0, 1.0)
  }

  return neuron
}

func (neuron *Neuron) Think (inputs []float64) float64 {
  out := neuron.bias;

  for i := 0; i < neuron.maxInputs; i++ {
    out += inputs[i] * neuron.weights[i]
    neuron.inputs[i] = inputs[i]
  }

  neuron.activation = 1.0 / (1.0 + math.Exp(-out))
  return neuron.activation
}

func NewLayer (maxNeurons int, maxInputs int) Layer {
  if maxInputs == 0 {
    if layerPrevNeurons == 0 {
      panic("need one layer with defined inputs")
    }

    maxInputs = layerPrevNeurons
  }
  
  layer := Layer{
    maxNeurons: maxNeurons,
    neurons: make([]Neuron, maxNeurons),
    inputMin: make([]float64, maxNeurons),
    inputMax: make([]float64, maxNeurons),
  }

  layerPrevNeurons = maxNeurons
  rand.Seed(time.Now().UnixNano())

  for i := 0; i < layer.maxNeurons; i++ {
    layer.neurons[i] = NewNeuron(maxInputs)
    layer.inputMin[i] = 0.0
    layer.inputMax[i] = 1.0
  }

  return layer
}

func (layer *Layer) Think (inputs []float64) []float64 {
  outs := make([]float64, layer.maxNeurons)

  for i := 0; i < layer.maxNeurons; i++ {
    outs[i] = layer.neurons[i].Think(inputs)
  }

  return outs
}

func NewNeural (layers []Layer) Neural {
  neural := Neural{
    maxLayers: len(layers),
    layers: layers,
  }

  return neural
}

func (neural *Neural) ThinkRaw (inputs []float64) []float64 {
  outs := neural.layers[0].Think(inputs)

  for i := 1; i < neural.maxLayers; i++ {
    outs = neural.layers[i].Think(outs)
  }

  return outs
}

func (neural *Neural) LearnRaw (inputs []float64, outputs []float64, rate float64) float64 {
  var mse float64
  outputLayer := &neural.layers[neural.maxLayers - 1]
  currentOut := neural.ThinkRaw(inputs)

  for o := 0; o < outputLayer.maxNeurons; o++ {
    output := &outputLayer.neurons[o]
    output.error = outputs[o] - currentOut[o]
    output.delta = output.activation * (1.0 - output.activation) * output.error

    mse += output.error * output.error
  }

  for l := neural.maxLayers - 2; l >= 0; l-- {
    layer := &neural.layers[l]
    nextLayer := &neural.layers[l + 1]

    for h := 0; h < layer.maxNeurons; h++ {
      hidden := &layer.neurons[h]
      hidden.error = 0.0

      for n := 0; n < nextLayer.maxNeurons; n++ {
        next := &nextLayer.neurons[n]
        hidden.error += next.weights[h] * next.delta

        for w := 0; w < next.maxInputs; w++ {
          next.weights[w] += next.inputs[w] * next.delta * rate
        }
        next.bias += next.delta * rate
      }

      hidden.delta = hidden.activation * (1.0 - hidden.activation) * hidden.error
    }
  }

  return mse / float64(outputLayer.maxNeurons)
}

// func (neural *Neural) Save () {}
// func (neural *Neural) Load () {}
// func (neural *Neural) Delete () {}
// func (neural *Neural) Reset () {}

func randomFloat(min float64, max float64) float64 {
  return min + rand.Float64() * (max - min)
}

// next methods are not actually needed but they are utilities for easy usage
func NewInputLayer (maxInputs int, maxNeurons int) Layer {
  return NewLayer(maxNeurons, maxInputs)
}

func NewHiddenLayer (maxNeurons int) Layer {
  return NewLayer(maxNeurons, 0)
}

func NewOutputLayer (maxNeurons int) Layer {
  return NewLayer(maxNeurons, 0)
}

func (neural *Neural) Think (inputs []float64) []float64 {
  return neural.OutputValuesFromRaw(neural.ThinkRaw(neural.InputValuesToRaw(inputs)))
}

func (neural *Neural) Learn (inputs []float64, outputs []float64, rate float64) float64 {
  return neural.LearnRaw(neural.InputValuesToRaw(inputs), neural.OutputValuesToRaw(outputs), rate)
}

func (neural *Neural) InputValuesToRaw (inputs []float64) []float64 {
  layer := neural.layers[0]
  raw := make([]float64, layer.neurons[0].maxInputs)

  for i := 0; i < layer.neurons[0].maxInputs; i++ {
    raw[i] = (inputs[i] - layer.inputMin[i]) / (layer.inputMax[i] - layer.inputMin[i])
  }

  return raw
}

func (neural *Neural) OutputValuesToRaw (outputs []float64) []float64 {
  layer := neural.layers[neural.maxLayers - 1]
  raw := make([]float64, layer.maxNeurons)

  for i := 0; i < layer.maxNeurons; i++ {
    raw[i] = (outputs[i] - layer.inputMin[i]) / (layer.inputMax[i] - layer.inputMin[i])
  }

  return raw
}

func (neural *Neural) OutputValuesFromRaw (outputs []float64) []float64 {
  layer := neural.layers[neural.maxLayers - 1]
  values := make([]float64, layer.maxNeurons)

  for i := 0; i < layer.maxNeurons; i++ {
    // inline rangeToRange
    values[i] = (layer.inputMax[i] - layer.inputMin[i]) / (1.0 - 0.0) * (outputs[i] - 1.0) + layer.inputMax[i]
  }

  return values
}

func (neural *Neural) InputRange (index int, min float64, max float64) {
  if neural.maxLayers == 0 {
    panic("need at least one layer created")
  }

  inputLayer := &neural.layers[0]
  inputLayer.inputMin[index] = min
  inputLayer.inputMax[index] = max
}

func (neural *Neural) OutputRange (index int, min float64, max float64) {
  if neural.maxLayers == 1 {
    panic("need at least two layers created")
  }

  outputLayer := &neural.layers[neural.maxLayers - 1]
  outputLayer.inputMin[index] = min
  outputLayer.inputMax[index] = max
}

func rangeToRange (v float64, fMin float64, fMax float64, tMin float64, tMax float64) float64 {
  return (tMax - tMin) / (fMax - fMin) * (v - fMax) + tMax
  /*
  more efficient with pre alloc
  a = (fMax - fMin) / (tMax - tMin)
  b = tMax - a * tMax
  then just
  return a * v + b
  */
}
