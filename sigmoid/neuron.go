package sigmoid

import (
	"math"
	"nn/util/vector"
)

type Sigmoid struct {
	numberOfInputs int
	values         [2]vector.Vector
	bias           float64
}

func NewSigmoid(numberOfInputs int) *Sigmoid {
	return &Sigmoid{
		numberOfInputs: numberOfInputs,
		values: [2]vector.Vector{
			*vector.NewVector(numberOfInputs),
			*vector.NewVector(numberOfInputs),
		},
	}
}

func (s *Sigmoid) SetBias(bias float64) {
	s.bias = bias
}

func (s *Sigmoid) SetWeights(weights []float64) {
	var n = len(weights)
	if n > s.numberOfInputs {
		n = s.numberOfInputs
	}

	for i := 0; i < n; i++ {
		s.values[0].Set(i, weights[i])
	}
}

func (s *Sigmoid) SetWeightAt(n int, weight float64) {
	s.values[0].Set(n, weight)
}

func (s *Sigmoid) SetInputs(inputs []float64) {
	var n = len(inputs)
	if n > s.numberOfInputs {
		n = s.numberOfInputs
	}

	for i := 0; i < n; i++ {
		s.values[1].Set(i, inputs[i])
	}
}

func (s *Sigmoid) SetInputAt(n int, input float64) {
	s.values[1].Set(n, input)
}

func (s Sigmoid) Value() float64 {
	v, ok := vector.DotProduct(&s.values[0], &s.values[1])
	if !ok {
		panic("DotProduct - vectors not same size")
	}

	return activationFunction(v + s.bias)
}

func activationFunction(v float64) float64 {
	return 1.0 / (1.0 + math.Exp(-v))
}
