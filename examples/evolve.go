package main

import (
	"fmt"
	"github.com/lukks/neural-go/v3"
	"time"
)

const fmtColor = "\033[0;36m%s\033[0m"

func main() {
	xor := neural.NewNeural([]*neural.Layer{
		{Inputs: 2, Units: 3},
		{Units: 3},
		{Units: 1, Loss: "mse"},
	})

	fmt.Printf(fmtColor, "think some values:\n")
	fmt.Printf("0, 0 [0] -> %f\n", xor.Think([]float64{0, 0}))
	fmt.Printf("1, 0 [1] -> %f\n", xor.Think([]float64{1, 0}))
	fmt.Printf("0, 1 [1] -> %f\n", xor.Think([]float64{0, 1}))
	fmt.Printf("1, 1 [0] -> %f\n", xor.Think([]float64{1, 1}))

	fmt.Printf(fmtColor, "learning:\n")
	start := millis()

	xor = xor.Evolve(neural.Evolve{
		Population: 20,
		Mutate:     0.05,
		Crossover:  0.5,
		Elitism:    5,
		Epochs:     100,
		Iterations: 50,
		Threshold:  0.00005,
		Dataset: [][][]float64{
			{{0, 0}, {0}},
			{{1, 0}, {1}},
			{{0, 1}, {1}},
			{{1, 1}, {0}},
		},
		Callback: func(epoch int, loss float64) bool {
			if epoch%10 == 0 || epoch == 99 {
				fmt.Printf("epoch=%v loss=%f elapsed=%v\n", epoch, loss, millis()-start)
			}

			return true
		},
	})

	fmt.Printf(fmtColor, "think some values:\n")
	fmt.Printf("0, 0 [0] -> %f\n", xor.Think([]float64{0, 0}))
	fmt.Printf("1, 0 [1] -> %f\n", xor.Think([]float64{1, 0}))
	fmt.Printf("0, 1 [1] -> %f\n", xor.Think([]float64{0, 1}))
	fmt.Printf("1, 1 [0] -> %f\n", xor.Think([]float64{1, 1}))

	fmt.Printf(fmtColor, "export:\n")
	json, _ := xor.Export()
	fmt.Printf("%s\n", json)
	// stream the json over network

	// or just xor.ToFile("./evolve.json")
}

func millis() int64 {
	return time.Now().UnixNano() / 1e6
}
