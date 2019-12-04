package neural

// LossFn is used to calculate the loss
type LossFn func (output float64, current float64) float64

