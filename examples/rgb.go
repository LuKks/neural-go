package main

import (
	"fmt"
	"github.com/lukks/neural-go"
)

const fmtColor = "\033[0;36m%s\033[0m"

func main() {
	rgb := neural.NewNeural([]*neural.Layer{
		{Inputs: 3, Units: 8, Range: [][]float64{{0, 255}, {0, 255}, {0, 255}}}, // r, g, b 0-255
		{Units: 8},
		{Units: 1, Range: [][]float64{{0, 100}}}, // brightness 0-100 (percentage)
	})

	fmt.Printf(fmtColor, "think some values:\n")
	fmt.Printf("255, 255, 255 [100] -> %f\n", rgb.Think([]float64{255, 255, 255}))
	fmt.Printf("0  , 0  , 0   [0] -> %f\n", rgb.Think([]float64{0, 0, 0}))

	fmt.Printf(fmtColor, "learning:\n")
	for i := 0; i <= 2000; i++ {
		light := []float64{100}
		dark := []float64{0}

		// LearnRaw doesn't use Range
		loss := rgb.LearnRaw([]float64{1, 0, 0}, []float64{1})
		loss += rgb.Learn([]float64{0, 255, 0}, light)
		loss += rgb.Learn([]float64{0, 0, 255}, light)
		loss += rgb.Learn([]float64{0, 0, 0}, dark)
		loss += rgb.Learn([]float64{100, 100, 100}, light)
		loss += rgb.Learn([]float64{107, 181, 255}, light)
		loss += rgb.Learn([]float64{0, 53, 105}, dark)
		loss += rgb.Learn([]float64{150, 150, 75}, light)
		loss += rgb.Learn([]float64{75, 75, 0}, dark)
		loss += rgb.Learn([]float64{0, 75, 75}, dark)
		loss += rgb.Learn([]float64{150, 74, 142}, light)
		loss += rgb.Learn([]float64{50, 50, 75}, dark)
		loss += rgb.Learn([]float64{103, 22, 94}, dark)
		loss /= 13

		if i%500 == 0 {
			fmt.Printf("iter %v, loss %f\n", i, loss)
		}
	}

	fmt.Printf(fmtColor, "think some values:\n")
	fmt.Printf("255, 255, 255 [100] -> %f\n", rgb.Think([]float64{255, 255, 255}))
	fmt.Printf("0  , 0  , 0   [0] -> %f\n", rgb.Think([]float64{0, 0, 0}))

	fmt.Printf(fmtColor, "think new values:\n")
	fmt.Printf("243, 179, 10  [100] -> %f\n", rgb.Think([]float64{243, 179, 10}))
	fmt.Printf("75 , 50 , 50  [0] -> %f\n", rgb.Think([]float64{75, 50, 50}))
	fmt.Printf("95 , 99 , 104 [100] -> %f\n", rgb.Think([]float64{95, 99, 104}))
	fmt.Printf("65 , 38 , 70  [0] -> %f\n", rgb.Think([]float64{65, 38, 70}))

	// ThinkRaw doesn't use Range
	fmt.Printf(fmtColor, "example using ThinkRaw:\n")
	fmt.Printf("65 , 38 , 70  [0.0] -> %f\n", rgb.ThinkRaw([]float64{0.254, 0.149, 0.274}))

}
