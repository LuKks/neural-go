package neural

import (
  "math"
)

// Forward to think
type ForwardFn func (sum float64) float64

// Backward to learn (derivative of forward)
type BackwardFn func (activation float64) float64

// Linear
func LinearForward (sum float64) float64 {
  return sum
}
// Linear derivative
func LinearBackward (activation float64) float64 {
  return 1.0
}

// Sigmoid
func SigmoidForward (sum float64) float64 {
  return 1.0 / (1.0 + math.Exp(-sum))
}
// Sigmoid derivative
func SigmoidBackward (activation float64) float64 {
  return activation * (1.0 - activation)
}

// Tanh
func TanhForward (sum float64) float64 {
  return math.Tanh(sum)
}
// Tanh derivative
func TanhBackward (activation float64) float64 {
  return 1 - activation * activation
}

// ReLU
func ReluForward (sum float64) float64 {
  if sum < 0.0 {
    return 0.0
  }
  return sum
}
// ReLU derivative
func ReluBackward (activation float64) float64 {
  if activation <= 0.0 {
    return 0.0
  }
  return 1.0
}

// Set of activations with its range
type ActivationSet struct {
  forward ForwardFn
  backward BackwardFn
  ranges []float64
}

func selectActivation (activation string) ActivationSet {
  set := ActivationSet{}

  if activation == "linear" {
    set.forward = LinearForward
    set.backward = LinearBackward
  } else if activation == "" || activation == "sigmoid" {
    set.forward = SigmoidForward
    set.backward = SigmoidBackward
    set.ranges = []float64{ 0.0, 1.0 }
  } else if activation == "tanh" {
    set.forward = TanhForward
    set.backward = TanhBackward
    set.ranges = []float64{ -1.0, 1.0 }
  } else if activation == "relu" {
    set.forward = ReluForward
    set.backward = ReluBackward
    set.ranges = []float64{ 0.0, 1.0 }
  } else {
    panic("need a valid activation name")
  }

  return set
}
