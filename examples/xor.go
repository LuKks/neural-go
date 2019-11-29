package main

import (
  "fmt"
  "github.com/lukks/neural-go"
)

func main () {
  rgb := neural.NewNeural([]neural.Layer{
    neural.NewLayer(8, 2), // NewLayer(neurons, inputs)
    neural.NewLayer(8, 0), // autorecognize previous layer, the 0 will be 8
    neural.NewLayer(1, 0), // due the inputs are the neurons of previous layer
  })

  fmt.Println("think some values:")
  fmt.Printf("0, 0 [0] -> %f\n", rgb.Think([]float64{ 0.0, 0.0 }))
  fmt.Printf("1, 0 [1] -> %f\n", rgb.Think([]float64{ 1.0, 0.0 }))
  fmt.Printf("0, 1 [1] -> %f\n", rgb.Think([]float64{ 0.0, 1.0 }))
  fmt.Printf("1, 1 [0] -> %f\n", rgb.Think([]float64{ 1.0, 1.0 }))

  fmt.Println("learning:")
  for i := 0; i <= 5000; i++ {
    rate := 0.4
    mse := rgb.LearnRaw([]float64{ 0.0, 0.0 }, []float64{ 0.0 }, rate)
    mse += rgb.LearnRaw([]float64{ 1.0, 0.0 }, []float64{ 1.0 }, rate)
    mse += rgb.LearnRaw([]float64{ 0.0, 1.0 }, []float64{ 1.0 }, rate)
    mse += rgb.LearnRaw([]float64{ 1.0, 1.0 }, []float64{ 0.0 }, rate)
    mse /= 4;

    if i % 1000 == 0 {
      fmt.Printf("iter %v, mse %f\n", i, mse)
    }
  }

  fmt.Println("think some values:")
  fmt.Printf("0, 0 [0] -> %f\n", rgb.Think([]float64{ 0.0, 0.0 }))
  fmt.Printf("1, 0 [1] -> %f\n", rgb.Think([]float64{ 1.0, 0.0 }))
  fmt.Printf("0, 1 [1] -> %f\n", rgb.Think([]float64{ 0.0, 1.0 }))
  fmt.Printf("1, 1 [0] -> %f\n", rgb.Think([]float64{ 1.0, 1.0 }))
}
