package main

import (
	"fmt"
)

type AnyNeuron struct {
	id int
}

type Neuron struct {
	AnyNeuron
	name string
	weights [16]float64
}

func main () {
	// float prec
	a := 0.1
	b := 0.2
	fmt.Println(a + b)

	// arrays
	var arr1 = [3]int{2, 4, 8}
	fmt.Println(arr1)

	var arr2 = [...]int{2, 4, 8}
	fmt.Println(arr2)

	// arr multi dim
	var arr3 [3][3]int = [3][3]int{ [3]int{1, 0, 0}, [3]int{0, 1, 0}, [3]int{0, 0, 1} }
	fmt.Println(arr3)

	var arr4 [3][3]int
	arr4[0] = [3]int{1, 0, 0}
	arr4[1] = [3]int{0, 1, 0}
	arr4[2] = [3]int{0, 0, 1}
	fmt.Println(arr4)

	// slices
	arr5 := []int{1, 2, 3}
	fmt.Println(len(arr5))
	fmt.Println(cap(arr5))

	// slices are by default referenced
	// arrays must use &

	// slice
	// res51 := arr5[:] // all
	// res52 := arr5[3:] // from 4th
	// res53 := arr5[:6] // to 6th

	// maps
	state := make(map[string]int)
	state = map[string]int{
		"abc": 111,
		"def": 222,
	}
	val, ok := state["ab"]
	fmt.Println(val, ok)

	neuron1 := Neuron{
		name: "n1",
	}
	fmt.Println(neuron1)

	neuron2 := struct{name string}{
		name: "n2",
	}
	fmt.Println(neuron2)
}
