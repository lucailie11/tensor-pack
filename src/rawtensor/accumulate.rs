use super::RawTensor;
use super::iter::LogicalIndices;
use super::structure::broadcast_shape;
use super::structure::expanded_strides;

use std::rc::Rc;

// In-place elementwise accumulation: self += f(a, b, c, ...) with broadcasting
// Used by grad/ to accumulate gradients
// Requires self to be the sole owner of its data (which is the case with gradients)

impl RawTensor {
    fn accumulate_n(&mut self, inputs: &[&RawTensor], f: impl Fn(&[f64]) -> f64) {
        let out_shape = inputs.iter().fold(
            self.shape.clone(),
            |acc, t| broadcast_shape(&acc, &t.shape).expect("shapes not broadcastable"),
        );

        let self_strides = expanded_strides(self, &out_shape).expect("self strides not expandable");
        let input_strides: Vec<Box<[usize]>> = inputs.iter()
            .map(|t| expanded_strides(t, &out_shape).expect("input strides not expandable"))
            .collect();

        let data = Rc::get_mut(&mut self.data).expect("couldn't borrow mutable data from the tensor");

        let mut iters: Vec<_> = input_strides.iter()
            .map(|s| LogicalIndices::new(out_shape.clone(), s.clone()))
            .collect();

        LogicalIndices::new(out_shape, self_strides).for_each(|i| {
            let vals: Vec<f64> = iters.iter_mut()
                .zip(inputs.iter())
                .map(|(it, t)| t.data[it.next().unwrap()])
                .collect();
            data[i] += f(&vals);
        });
    }

    pub(crate) fn accumulate_1(&mut self, a: &RawTensor, f: impl Fn(f64) -> f64) {
        self.accumulate_n(&[a], |v| f(v[0]));
    }

    pub(crate) fn accumulate_2(&mut self, a: &RawTensor, b: &RawTensor, f: impl Fn(f64, f64) -> f64) {
        self.accumulate_n(&[a, b], |v| f(v[0], v[1]));
    }

    pub(crate) fn accumulate_3(&mut self, a: &RawTensor, b: &RawTensor, c: &RawTensor, f: impl Fn(f64, f64, f64) -> f64) {
        self.accumulate_n(&[a, b, c], |v| f(v[0], v[1], v[2]));
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let mut grad = RawTensor::zeros(&[3]);
        grad.accumulate_1(&RawTensor::linspace(1.0, 3.0, 3), |gv| gv * 2.0);
        assert_eq!(*grad.contiguous_data(), [2.0, 4.0, 6.0]);

        let mut grad = RawTensor::zeros(&[3]);
        grad.accumulate_2(&RawTensor::ones(&[3]), &RawTensor::from_slice(&[3], &[2.0, 4.0, 8.0]), |gv, bv| gv / bv);
        assert_eq!(*grad.contiguous_data(), [0.5, 0.25, 0.125]);
    }

    #[test]
    fn accumulate_3_basic() {
        let g = RawTensor::ones(&[3]);
        let a = RawTensor::from_slice(&[3], &[6.0, 4.0, 9.0]);
        let b = RawTensor::from_slice(&[3], &[2.0, 2.0, 3.0]);
        let mut grad = RawTensor::zeros(&[3]);
        grad.accumulate_3(&g, &a, &b, |gv, av, bv| gv * (-av) / (bv * bv));
        assert_eq!(*grad.contiguous_data(), [-1.5, -1.0, -1.0]);
        grad.accumulate_3(&g, &a, &b, |gv, av, bv| gv * (-av) / (bv * bv));
        assert_eq!(*grad.contiguous_data(), [-3.0, -2.0, -2.0]);
    }

    #[test]
    fn accumulate_3_broadcast() {
        let mut grad = RawTensor::zeros(&[3, 3]);
        let g = RawTensor::ones(&[1, 3]);
        let a = RawTensor::linspace(1.0, 3.0, 3).reshape(&[3, 1]);
        let b = RawTensor::linspace(1.0, 3.0, 3).reshape(&[1, 3]);
        grad.accumulate_3(&g, &a, &b, |gv, av, bv| gv * av * bv);
        assert_eq!(*grad.contiguous_data(), [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn accumulate_3_transpose_expand() {
        let a = RawTensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).transpose(&[1, 0]);
        let b = RawTensor::from_slice(&[1, 2], &[1.0, 2.0]).expand(&[3, 2]);
        let g = RawTensor::ones(&[3, 2]);
        let mut grad = RawTensor::zeros(&[3, 2]);
        grad.accumulate_3(&g, &a, &b, |gv, av, bv| gv * av * bv);
        assert_eq!(*grad.contiguous_data(), [1.0, 8.0, 2.0, 10.0, 3.0, 12.0]);
        grad.accumulate_3(&g, &a, &b, |gv, av, bv| gv * av * bv);
        assert_eq!(*grad.contiguous_data(), [2.0, 16.0, 4.0, 20.0, 6.0, 24.0]);
    }
}
