# neural-go

Neural networks (deep feedforward)

![](https://img.shields.io/github/v/release/LuKks/neural-go) [![](https://img.shields.io/maintenance/yes/2019.svg?style=flat-square)](https://github.com/LuKks/neural-go) ![](https://img.shields.io/github/size/LuKks/neural-go/index.go.svg) ![](https://img.shields.io/github/license/LuKks/neural-go.svg)

```golang
package main

import (
  "fmt"
  "github.com/lukks/neural-go"
)

func main () {
  rgb := neural.NewNeural([]neural.Layer{
    neural.NewInputLayer(2, 8), // (inputs, neurons) is also a hidden layer
    neural.NewHiddenLayer(8),
    neural.NewOutputLayer(1),
  })

  for i := 0; i <= 5000; i++ {
    mse := rgb.LearnRaw([]float64{ 0.0, 0.0 }, []float64{ 0.0 }, 0.2)
    mse += rgb.LearnRaw([]float64{ 1.0, 0.0 }, []float64{ 1.0 }, 0.2)
    mse += rgb.LearnRaw([]float64{ 0.0, 1.0 }, []float64{ 1.0 }, 0.2)
    mse += rgb.LearnRaw([]float64{ 1.0, 1.0 }, []float64{ 0.0 }, 0.2)
    mse /= 4;

    if i % 1000 == 0 {
      fmt.Printf("iter %v, mse %f\n", i, mse)
    }
  }

  fmt.Printf("0, 0 [0] -> %f\n", rgb.Think([]float64{ 0.0, 0.0 }))
  fmt.Printf("1, 0 [1] -> %f\n", rgb.Think([]float64{ 1.0, 0.0 }))
  fmt.Printf("0, 1 [1] -> %f\n", rgb.Think([]float64{ 0.0, 1.0 }))
  fmt.Printf("1, 1 [0] -> %f\n", rgb.Think([]float64{ 1.0, 1.0 }))
}
```

## Install
```
go get github.com/lukks/neural-go
```

## Features
#### Ranges
Set a range of values for every input and output.\
So you use your values as you know but the neural get the values in raw (0-1).\
Check [examples/full.go](https://github.com/LuKks/neural-go/blob/master/examples/full.go) for usage example.

#### Description
From my previous [neural-amxx](https://github.com/LuKks/neural-amxx).

#### Structs
```golang
type Neuron struct
type Layer struct
type Neural struct
```

#### Methods
```golang
// low-level usage:
func NewNeuron (maxInputs int) Neuron {}
func (neuron *Neuron) Think (inputs []float64) float64 {}

func NewLayer (maxNeurons int, maxInputs int) Layer {}
func (layer *Layer) Think (inputs []float64) []float64 {}

func NewNeural (layers []Layer) Neural {}
func (neural *Neural) ThinkRaw (inputs []float64) []float64 {}
func (neural *Neural) LearnRaw (inputs []float64, outputs []float64, rate float64) float64 {}

// high-level usage:
func NewInputLayer (maxInputs int, maxNeurons int) Layer {}
func NewHiddenLayer (maxNeurons int) Layer {}
func NewOutputLayer (maxNeurons int) Layer {}
func (neural *Neural) InputRange (index int, min float64, max float64) {}
func (neural *Neural) OutputRange (index int, min float64, max float64) {}
func (neural *Neural) Think (inputs []float64) []float64 {}
func (neural *Neural) Learn (inputs []float64, outputs []float64, rate float64) float64 {}
```

#### Missing
Some methods that are not available yet:
```golang
func (neural *Neural) Save () {}
func (neural *Neural) Load () {}
func (neural *Neural) Delete () {}
func (neural *Neural) Reset () {}
```

## Examples
Basic XOR [examples/xor.go](https://github.com/LuKks/neural-go/blob/master/examples/xor.go)\
RGB brightness [examples/full.go](https://github.com/LuKks/neural-go/blob/master/examples/full.go)

```
go run examples/full.go
```

## Tests
```
There are no tests yet
```

## Issues
Feedback, ideas, etc are very welcome so feel free to open an issue.

## License
Code released under the [MIT License](https://github.com/LuKks/neural-go/blob/master/LICENSE).
