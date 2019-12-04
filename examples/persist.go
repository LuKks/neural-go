package main

import (
	"fmt"
	"github.com/lukks/neural-go"
)

const fmtColor = "\033[0;36m%s\033[0m"

func main() {
	xor := neural.NewNeural([]*neural.Layer{
		{Inputs: 2, Units: 3},
		{Units: 3},
		{Units: 1},
	})

	fmt.Printf(fmtColor, "learning:\n")
	for i := 0; i <= 50000; i++ {
		loss := xor.Learns([][][]float64{
			{{0, 0}, {0}},
			{{1, 0}, {1}},
			{{0, 1}, {1}},
			{{1, 1}, {0}},
		})

		if i%10000 == 0 {
			fmt.Printf("iter %v, loss %f\n", i, loss)
		}
	}

	fmt.Printf(fmtColor, "think some values:\n")
	fmt.Printf("0, 0 [0] -> %f\n", xor.Think([]float64{0, 0}))
	fmt.Printf("1, 0 [1] -> %f\n", xor.Think([]float64{1, 0}))
	fmt.Printf("0, 1 [1] -> %f\n", xor.Think([]float64{0, 1}))
	fmt.Printf("1, 1 [0] -> %f\n", xor.Think([]float64{1, 1}))

	fmt.Printf(fmtColor, "to file:\n")
	err1 := xor.ToFile("./evolve.json")
	fmt.Printf("err? %v\n", err1)

	fmt.Printf(fmtColor, "from file:\n")
	err2 := xor.FromFile("./evolve.json")
	fmt.Printf("err? %v\n", err2)

	fmt.Printf(fmtColor, "delete file:\n")
	err3 := xor.DeleteFile("./evolve.json")
	fmt.Printf("err? %v\n", err3)

	fmt.Printf(fmtColor, "to string (export):\n")
	json, _ := xor.Export()
	fmt.Printf("%s\n", json)

	// can stream the exported json over network
	// then in a different process/machine:
	fmt.Printf(fmtColor, "from string (import):\n")
	xorCopy := neural.NewNeural([]*neural.Layer{})
	err4 := xorCopy.Import(json)
	fmt.Printf("err? %v\n", err4)
}
