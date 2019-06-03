package vector

import "math"

type Vector struct {
	buffer []float64
	n      int
}

func NewVector(size int) *Vector {
	return &Vector{
		buffer: make([]float64, size),
		n:      size,
	}
}

func (v *Vector) Init(vals []float64) {
	var n = v.n
	if n > len(vals) {
		n = len(vals)
	}

	for i := 0; i < n; i++ {
		v.buffer[i] = vals[i]
	}
}

func (v *Vector) Set(i int, val float64) {
	v.buffer[i] = val
}

func (v *Vector) Get(i int) float64 {
	return v.buffer[i]
}

func (v *Vector) Subtract(a *Vector) {
	l := v.n
	if l > a.n {
		l = a.n
	}

	for i := 0; i < l; i++ {
		v.buffer[i] -= a.buffer[i]
	}
}

func (v *Vector) Add(a *Vector) {
	l := v.n
	if l > a.n {
		l = a.n
	}

	for i := 0; i < l; i++ {
		v.buffer[i] += a.buffer[i]
	}
}

func Subtraction(a, b *Vector) (*Vector, bool) {
	if a.n != b.n {
		return nil, false
	}

	v := NewVector(a.n)

	for i := 0; i < a.n; i++ {

		v.buffer[i] = a.buffer[i] - b.buffer[i]
	}

	return v, true
}

func Addition(a, b *Vector) (*Vector, bool) {
	if a.n != b.n {
		return nil, false
	}

	v := NewVector(a.n)

	for i := 0; i < a.n; i++ {

		v.buffer[i] = a.buffer[i] + b.buffer[i]
	}

	return v, true
}

func DotProduct(a, b *Vector) (float64, bool) {
	if a.n != b.n {
		return 0.0, false
	}

	var sum = 0.0
	for i := 0; i < a.n; i++ {

		sum += (a.buffer[i] * b.buffer[i])
	}

	return sum, true
}

func Magnitude(a, b *Vector) (float64, bool) {
	dp, ok := DotProduct(a, b)
	if !ok {
		return dp, ok
	}

	return math.Sqrt(dp), ok
}
