package main

import (
	"fmt"
	"github.com/lukks/neural-go/v3"
)

const fmtColor = "\033[0;36m%s\033[0m"

func main() {
	xor := neural.NewNeural([]*neural.Layer{
		{Inputs: 2, Units: 16, Activation: "sigmoid", Rate: 0.002, Momentum: 0.999},
		{Units: 16, Activation: "tanh", Rate: 0.001},
		{Units: 1, Activation: "sigmoid", Loss: "mse", Rate: 0.0005},
	})
	// that is just to show different configurations
	// normally you want same rate and momentum for all layers

	/*
	  // Change rate or momentum to all layers
	  xor.Rate(0.002)
	  xor.Momentum(0.999)

	  // Change to specific layer
	  xor.Layers[0].Rate = 0.002
	  xor.Layers[0].Momentum = 0.999
	*/

	fmt.Printf(fmtColor, "learning:\n")
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

	fmt.Printf(fmtColor, "think some values:\n")
	fmt.Printf("0, 0 [0] -> %f\n", xor.Think([]float64{0, 0}))
	fmt.Printf("1, 0 [1] -> %f\n", xor.Think([]float64{1, 0}))
	fmt.Printf("0, 1 [1] -> %f\n", xor.Think([]float64{0, 1}))
	fmt.Printf("1, 1 [0] -> %f\n", xor.Think([]float64{1, 1}))
}
