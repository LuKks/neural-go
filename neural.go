package neural

import (
  "sort"
  "fmt"
  "encoding/json"
  "io/ioutil"
  "os"
)

// Neural is a set of layers
type Neural struct {
  // Max layers in neural
  MaxLayers int `json:"-"`
  // Slice of layers
  Layers []*Layer `json:"Layers"`
  // Average of loss (used in Learns and Evolve)
  Loss float64 `json:"-"`
}

// Evolve is the config for evolution process
type Evolve struct {
  Population int
  Mutate float64
  Crossover float64
  Elitism int
  Epochs int
  Iterations int
  Threshold float64
  Dataset [][][]float64
  Callback func (epoch int, loss float64) bool
}

// NewNeural creates a neural based on multiple layers
func NewNeural (layers []*Layer) *Neural {
  fmt.Printf("")

  neural := &Neural{
    MaxLayers: len(layers),
    Layers: make([]*Layer, len(layers)),
  }

  for i, prevUnits := 0, 0; i < neural.MaxLayers; i++ {
    if layers[i].Inputs == 0 {
      if prevUnits == 0 {
        panic("need the first layer with defined inputs")
      }

      layers[i].Inputs = prevUnits
    }

    prevUnits = layers[i].Units
    neural.Layers[i] = NewLayer(layers[i])
  }

  return neural
}

// ThinkRaw process the neural forward based on inputs and then based on output of previous layer 
func (neural *Neural) ThinkRaw (inputs []float64) []float64 {
  outs := neural.Layers[0].Think(inputs)

  for i := 1; i < neural.MaxLayers; i++ {
    outs = neural.Layers[i].Think(outs)
  }

  return outs
}

// Think arbitrary values by automatic conversion to raw values and vice versa for output
func (neural *Neural) Think (inputs []float64) []float64 {
  return neural.OutputValuesFromRaw(neural.ThinkRaw(neural.InputValuesToRaw(inputs)))
}

// LearnRaw uses backpropagation
func (neural *Neural) LearnRaw (inputs []float64, outputs []float64) float64 {
  loss := 0.0
  outputLayer := neural.Layers[neural.MaxLayers - 1]
  currentOut := neural.ThinkRaw(inputs)

  for o, output := range outputLayer.Neurons {
    output.error = outputs[o] - currentOut[o]
    output.delta = output.Layer.Backward(output.activation) * output.error
    loss += output.error * output.error
  }

  for l := neural.MaxLayers - 2; l >= 0; l-- {
    layer := neural.Layers[l]
    nextLayer := neural.Layers[l + 1]

    for h, hidden := range layer.Neurons {
      hidden.error = 0.0
      for _, next := range nextLayer.Neurons {
        hidden.error += next.Weights[h] * next.delta
      }
      hidden.delta = hidden.Layer.Backward(hidden.activation) * hidden.error
    }

    for _, next := range nextLayer.Neurons {
      for w := 0; w < next.MaxInputs; w++ {
        next.Weights[w] += next.Optimizer(w, next.Inputs[w] * next.delta * next.Layer.Rate)
      }
      next.Bias += next.Optimizer(next.MaxInputs, next.delta * next.Layer.Rate)
    }
  }

  return loss / float64(outputLayer.Units)
}

// LearnsRaw is a shorcut to learn a raw dataset of inputs/outputs backed by LearnRaw method
func (neural *Neural) LearnsRaw (dataset [][][]float64) float64 {
  neural.Loss = 0.0
  for _, data := range dataset {
    neural.Loss += neural.LearnRaw(data[0], data[1])
  }
  neural.Loss /= float64(len(dataset))
  return neural.Loss
}

// Learn arbitrary values by automatic conversion to raw values
func (neural *Neural) Learn (inputs []float64, outputs []float64) float64 {
  return neural.LearnRaw(neural.InputValuesToRaw(inputs), neural.OutputValuesToRaw(outputs))
}

// Learns is a shorcut to learn dataset of arbitrary inputs/outputs backed by Learn method
func (neural *Neural) Learns (dataset [][][]float64) float64 {
  neural.Loss = 0.0
  for _, data := range dataset {
    neural.Loss += neural.Learn(data[0], data[1])
  }
  neural.Loss /= float64(len(dataset))
  return neural.Loss
}

// Clone neural with same layers
func (neural *Neural) Clone () *Neural {
  layers := make([]*Layer, neural.MaxLayers)

  for i := 0; i < neural.MaxLayers; i++ {
    layers[i] = &Layer{
      Inputs: neural.Layers[i].Inputs,
      Units: neural.Layers[i].Units,
      Activation: neural.Layers[i].Activation,
      Rate: neural.Layers[i].Rate,
      Momentum: neural.Layers[i].Momentum,
    }

    layers[i].Range = make([][]float64, len(neural.Layers[i].Range))
    copy(layers[i].Range, neural.Layers[i].Range)
  }

  clone := NewNeural(layers)

  for i := 0; i < neural.MaxLayers; i++ {
    clone.Layers[i] = neural.Layers[i].Clone()
  }

  return clone
}

// Mutate neurons of all layers based on probability
func (neural *Neural) Mutate (probability float64) {
  for i := 0; i < neural.MaxLayers; i++ {
    neural.Layers[i].Mutate(probability)
  }
}

// Crossover two neurals merging layers
func (neural *Neural) Crossover (neuralB *Neural, dominant float64) *Neural {
  new := NewNeural([]*Layer{})
  new.MaxLayers = neural.MaxLayers
  new.Layers = make([]*Layer, neural.MaxLayers)

  for i := 0; i < neural.MaxLayers; i++ {
    new.Layers[i] = neural.Layers[i].Crossover(neuralB.Layers[i], dominant)
  }

  return new
}

// Evolve uses Clone, Mutate, Learns and Crossover to create a evolutionary scenario
func (neural *Neural) Evolve (evolve Evolve) *Neural {
  if evolve.Population == 0 {
    evolve.Population = 20
  }
  if evolve.Mutate == 0.0 {
    evolve.Mutate = 0.01
  }
  if evolve.Crossover == 0.0 {
    evolve.Crossover = 0.5
  }
  if evolve.Elitism == 0 {
    evolve.Elitism = 5
  }
  if evolve.Epochs == 0 {
    panic("need to set epochs in evolve")
  }
  if evolve.Iterations == 0 {
    evolve.Iterations = 1
  }

  population := make([]*Neural, evolve.Population)
  for p := 0; p < evolve.Population; p++ {
    population[p] = neural.Clone()
  }

  for e := 0; e < evolve.Epochs; e++ {
    for p := 0; p < evolve.Population; p++ {
      population[p].Mutate(evolve.Mutate)

      for i := 0; i < evolve.Iterations; i++ {
        population[p].Learns(evolve.Dataset)
      }
    }

    sort.Slice(population, func (a int, b int) bool {
      return population[a].Loss < population[b].Loss
    })

    if evolve.Callback(e, population[0].Loss) == false {
      break
    }

    if population[0].Loss <= evolve.Threshold {
      break
    }

    if e == evolve.Epochs - 1 {
      break
    }

    for p := 0; p < evolve.Population; p++ {
      if p < evolve.Elitism {
        randomIndex := randomInt(int64(evolve.Population))
        children := population[p].Crossover(population[randomIndex], evolve.Crossover)

        population[evolve.Population - 1 - p] = children
      }
    }
  }

  return population[0]
}


// Reset neurons (weights, bias, etc) of all layers
func (neural *Neural) Reset () {
  for i := 0; i < neural.MaxLayers; i++ {
    neural.Layers[i].Reset()
  }
}

// Rate set the rate for all layers
func (neural *Neural) Rate (value float64) {
  for i := 0; i < neural.MaxLayers; i++ {
    neural.Layers[i].Rate = value
  }
}

// Momentum set the momentum for all layers
func (neural *Neural) Momentum (value float64) {
  for i := 0; i < neural.MaxLayers; i++ {
    neural.Layers[i].Momentum = value
  }
}

// Export neural to json string
func (neural *Neural) Export () ([]byte, error) {
  return json.Marshal(&neural)
}

// Import neural from json string
func (neural *Neural) Import (encoded []byte) error {
  json.Unmarshal(encoded, &neural)

  neural.MaxLayers = len(neural.Layers)

  for _, layer := range neural.Layers {
    layer.Inputs = len(layer.Neurons[0].Weights)
    layer.Units = len(layer.Neurons)
    layer.SetActivation(layer.Activation)

    for _, neuron := range layer.Neurons {
      neuron.MaxInputs = len(neuron.Weights)
      neuron.Layer = layer
      neuron.Inputs = make([]float64, neuron.MaxInputs)
    }
  }

  return nil
}

// ToFile export neural to file
func (neural *Neural) ToFile (filename string) error {
  encoded, err := neural.Export()
  if err != nil {
    return err
  }
  return ioutil.WriteFile(filename, encoded, 0644)
}

// FromFile import neural from file
func (neural *Neural) FromFile (filename string) error {
  content, err := ioutil.ReadFile(filename)
  if err != nil {
    return err
  }
  return neural.Import(content)
}

// DeleteFile is a shortcut to delete a file
func (neural *Neural) DeleteFile (filename string) error {
  return os.Remove(filename)
}

// InputValuesToRaw converts arbitrary input values to raw (using layer range property)
func (neural *Neural) InputValuesToRaw (inputs []float64) []float64 {
  layer := neural.Layers[0]
  total := len(layer.Range)
  if total == 0 {
    return inputs
  }

  raw := make([]float64, total)
  for i, ranges := range layer.Range {
    raw[i] = rangeToRange(inputs[i], ranges[0], ranges[1], ranges[2], ranges[3])
  }
  return raw
}

// OutputValuesToRaw converts arbitrary output values to raw (using layer range property)
func (neural *Neural) OutputValuesToRaw (outputs []float64) []float64 {
  layer := neural.Layers[neural.MaxLayers - 1]
  total := len(layer.Range)
  if total == 0 {
    return outputs
  }

  raw := make([]float64, total)
  for i, ranges := range layer.Range {
    raw[i] = rangeToRange(outputs[i], ranges[0], ranges[1], ranges[2], ranges[3])
  }
  return raw
}

// OutputValuesFromRaw converts raw output to arbitrary output values (using layer range property)
func (neural *Neural) OutputValuesFromRaw (outputs []float64) []float64 {
  layer := neural.Layers[neural.MaxLayers - 1]
  total := len(layer.Range)
  if total == 0 {
    return outputs
  }

  values := make([]float64, total)
  for i, ranges := range layer.Range {
    values[i] = rangeToRange(outputs[i], ranges[2], ranges[3], ranges[0], ranges[1])
  }
  return values
}

func rangeToRange (v float64, fMin float64, fMax float64, tMin float64, tMax float64) float64 {
  return (tMax - tMin) / (fMax - fMin) * (v - fMax) + tMax
  /*
  more efficient with pre alloc
  x = (tMax - tMin) / (fMax - fMin)
  y = x * fMax - tMax
  then just
  return v * x + y
  */
}
