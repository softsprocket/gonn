package main

import (
	"fmt"
	"nn/perceptron"
	"nn/sigmoid"
)

func main() {

	p := perceptron.NewPerceptron(2, func(f float64) bool { return f > 0 })

	p.SetWeights([]float64{-2.0, -2.0})
	p.SetBias(3.0)

	p.SetInputs([]float64{1.0, 1.0})
	result := p.Value()
	fmt.Printf("%f %t\n", result, p.Fire())

	p.SetInputs([]float64{0.0, 0.0})
	result = p.Value()
	fmt.Printf("%f %t\n", result, p.Fire())

	s := sigmoid.NewSigmoidNeuron(2)

	s.SetWeights([]float64{-2.0, -2.0})
	s.SetBias(3.0)

	s.SetInputs([]float64{1.0, 1.0})
	result = s.Value()
	fmt.Printf("%f\n", result)

	s.SetInputs([]float64{0.0, 0.0})
	result = s.Value()
	fmt.Printf("%f\n", result)
}
