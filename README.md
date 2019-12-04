# neural-go

Genetic Neural Networks

[![Go Report Card](https://goreportcard.com/badge/github.com/LuKks/neural-go?t=0)](https://goreportcard.com/report/github.com/LuKks/neural-go) ![](https://img.shields.io/github/v/release/LuKks/neural-go) [![GoDoc](https://godoc.org/github.com/LuKks/neural-go?status.svg)](https://godoc.org/github.com/LuKks/neural-go) ![](https://img.shields.io/github/license/LuKks/neural-go.svg)

```golang
package main

import (
  "fmt"
  "github.com/lukks/neural-go"
)

func main() {
  xor := neural.NewNeural([]*neural.Layer{
    {Inputs: 2, Units: 16},
    {Units: 16},
    {Units: 1},
  })

  for i := 0; i <= 5000; i++ {
    loss := xor.Learns([][][]float64{
      {{0, 0}, {0}},
      {{1, 0}, {1}},
      {{0, 1}, {1}},
      {{1, 1}, {0}},
    })

    if i%1000 == 0 {
      fmt.Printf("iter %v, loss %f\n", i, loss)
    }
  }

  fmt.Printf("think some values:\n")
  fmt.Printf("0, 0 [0] -> %f\n", xor.Think([]float64{0, 0}))
  fmt.Printf("1, 0 [1] -> %f\n", xor.Think([]float64{1, 0}))
  fmt.Printf("0, 1 [1] -> %f\n", xor.Think([]float64{0, 1}))
  fmt.Printf("1, 1 [0] -> %f\n", xor.Think([]float64{1, 1}))
}
```

## Install latest version
```
go get github.com/lukks/neural-go
```

Also find versions on [releases](https://github.com/LuKks/neural-go/releases).\
So you can stick with a specific version: `go get gopkg.in/lukks/neural-go.v2`

## Features
#### Range
Set a range of values for every input and output.\
So you use your values as you know but the neural get it in raw activation.\
Check [examples/rgb.go](https://github.com/LuKks/neural-go/blob/master/examples/rgb.go) for usage example.

#### Customizable
Set different activations, rates, momentums, etc at layer level.
- Activation: `linear`, `sigmoid` (default), `tanh` and `relu`
- Learning Rate
- Optimizer by Momentum
- Loss: for output layer, only `mse` for now
- Range: for input and output layer

Check [examples/layers.go](https://github.com/LuKks/neural-go/blob/master/examples/layers.go) for complete example.

#### Genetics
Clone, mutate and crossover neurons, layers and neurals.\
The `Evolve` method internally uses these methods to put this very easy.\
Check [examples/evolve.go](https://github.com/LuKks/neural-go/blob/master/examples/evolve.go) but it's optional, not always need to use genetics.

#### Utils
There are several useful methods: Export, Import, Reset, ToFile, FromFile, etc.\
Check the [documentation here](https://godoc.org/github.com/LuKks/neural-go).

#### Description
From my previous [neural-amxx](https://github.com/LuKks/neural-amxx).

## Examples
Basic XOR [examples/xor.go](https://github.com/LuKks/neural-go/blob/master/examples/xor.go)\
RGB brightness [examples/rgb.go](https://github.com/LuKks/neural-go/blob/master/examples/rgb.go)\
Genetics [examples/evolve.go](https://github.com/LuKks/neural-go/blob/master/examples/evolve.go)\
Layer configs [examples/layers.go](https://github.com/LuKks/neural-go/blob/master/examples/layers.go)\
Persist [examples/persist.go](https://github.com/LuKks/neural-go/blob/master/examples/persist.go)

```
go run examples/rgb.go
```

## Tests
```
There are no tests yet
```

## Issues
Feedback, ideas, etc are very welcome so feel free to open an issue.

## License
Code released under the [MIT License](https://github.com/LuKks/neural-go/blob/master/LICENSE).
