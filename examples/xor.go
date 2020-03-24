package main

import (
	"fmt"
	"github.com/lukks/neural-go/v3"
)

const fmtColor = "\033[0;36m%s\033[0m"

func main() {
	xor := neural.NewNeural([]*neural.Layer{
		{Inputs: 2, Units: 16}, // input
		{Units: 16},            // hidden/s
		{Units: 1},             // output
	})

	fmt.Printf(fmtColor, "think some values:\n")
	fmt.Printf("0, 0 [0] -> %f\n", xor.Think([]float64{0, 0}))
	fmt.Printf("1, 0 [1] -> %f\n", xor.Think([]float64{1, 0}))
	fmt.Printf("0, 1 [1] -> %f\n", xor.Think([]float64{0, 1}))
	fmt.Printf("1, 1 [0] -> %f\n", xor.Think([]float64{1, 1}))

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
