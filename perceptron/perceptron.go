package perceptron

import (
	"nn/util/vector"
)

type ThresholdFunction func(float64) bool

type Perceptron struct {
	numberOfInputs    int
	values            [2]vector.Vector
	bias              float64
	thresholdFunction ThresholdFunction
}

func NewPerceptron(numberOfInputs int, f ThresholdFunction) *Perceptron {
	return &Perceptron{
		numberOfInputs: numberOfInputs,
		values: [2]vector.Vector{
			*vector.NewVector(numberOfInputs),
			*vector.NewVector(numberOfInputs),
		},
		thresholdFunction: f,
	}
}

func (p *Perceptron) SetBias(bias float64) {
	p.bias = bias
}

func (p *Perceptron) SetWeights(weights []float64) {
	var n = len(weights)
	if n > p.numberOfInputs {
		n = p.numberOfInputs
	}

	for i := 0; i < n; i++ {
		p.values[0].Set(i, weights[i])
	}
}

func (p *Perceptron) SetWeightAt(n int, weight float64) {
	p.values[0].Set(n, weight)
}

func (p *Perceptron) SetInputs(inputs []float64) {
	var n = len(inputs)
	if n > p.numberOfInputs {
		n = p.numberOfInputs
	}

	for i := 0; i < n; i++ {
		p.values[1].Set(i, inputs[i])
	}
}

func (p *Perceptron) SetInputAt(n int, input float64) {
	p.values[1].Set(n, input)
}

func (p Perceptron) Value() float64 {
	v, ok := vector.DotProduct(&p.values[0], &p.values[1])
	if !ok {
		panic("DotProduct - vectors not same size")
	}

	return v + p.bias
}

func (p Perceptron) Fire() bool {
	return p.thresholdFunction(p.Value())
}
