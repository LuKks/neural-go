package neural

import (
	"math"
)

// ForwardFn is used to think
type ForwardFn func(sum float64) float64

// BackwardFn is used to learn (derivative of forward)
type BackwardFn func(activation float64) float64

// LinearForward is the linear fn
func LinearForward(sum float64) float64 {
	return sum
}

// LinearBackward is the linear derivative
func LinearBackward(activation float64) float64 {
	return 1.0
}

// SigmoidForward is the sigmoid fn
func SigmoidForward(sum float64) float64 {
	return 1.0 / (1.0 + math.Exp(-sum))
}

// SigmoidBackward is the sigmoid derivative
func SigmoidBackward(activation float64) float64 {
	return activation * (1.0 - activation)
}

// TanhForward is the tanh fn
func TanhForward(sum float64) float64 {
	return math.Tanh(sum)
}

// TanhBackward is the tanh derivative
func TanhBackward(activation float64) float64 {
	return 1 - activation*activation
}

// ReluForward is the relu fn
func ReluForward(sum float64) float64 {
	if sum < 0.0 {
		return 0.0
	}
	return sum
}

// ReluBackward is the relu derivative
func ReluBackward(activation float64) float64 {
	if activation <= 0.0 {
		return 0.0
	}
	return 1.0
}

// ActivationSet is a forward and backward fn with its range
type ActivationSet struct {
	// Forward fn
	Forward ForwardFn
	// Backward fn
	Backward BackwardFn
	// Range of the activation
	Ranges []float64
}

func selectActivation(activation string) ActivationSet {
	set := ActivationSet{}

	if activation == "linear" {
		set.Forward = LinearForward
		set.Backward = LinearBackward
	} else if activation == "" || activation == "sigmoid" {
		set.Forward = SigmoidForward
		set.Backward = SigmoidBackward
		set.Ranges = []float64{0.0, 1.0}
	} else if activation == "tanh" {
		set.Forward = TanhForward
		set.Backward = TanhBackward
		set.Ranges = []float64{-1.0, 1.0}
	} else if activation == "relu" {
		set.Forward = ReluForward
		set.Backward = ReluBackward
		set.Ranges = []float64{0.0, 1.0}
	} else {
		panic("need a valid activation name")
	}

	return set
}
