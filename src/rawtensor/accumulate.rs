use super::RawTensor;
use super::iter::LogicalIndices;
use super::structure::broadcast_shape;
use super::structure::expanded_strides;

use std::rc::Rc;

// In-place elementwise accumulation: self += f(a, b, c, ...) with broadcasting
// Used by grad/ to accumulate gradients
// Requires self to be the sole owner of its data (which is the case with gradients)

impl RawTensor {
    pub(crate) fn accumulate_1(&mut self, a: &RawTensor, f: impl Fn(f64) -> f64) {
        let out_shape: Box<[usize]> = broadcast_shape(&self.shape, &a.shape);

        let self_strides: Box<[usize]> = expanded_strides(self, &out_shape);
        let a_strides: Box<[usize]> = expanded_strides(a, &out_shape);

        let data = Rc::get_mut(&mut self.data).expect("couldn't borrow mutable data from the tensor");

        LogicalIndices::new(out_shape.clone(), self_strides)
            .zip(LogicalIndices::new(out_shape.clone(), a_strides))
            .for_each(|(i, j)| data[i] += f(a.data[j]));
    }

    pub(crate) fn accumulate_2(&mut self, a: &RawTensor, b: &RawTensor, f: impl Fn(f64, f64) -> f64) {
        let out_shape: Box<[usize]> = broadcast_shape(&self.shape, &broadcast_shape(&a.shape, &b.shape));

        let self_strides: Box<[usize]> = expanded_strides(self, &out_shape);
        let a_strides: Box<[usize]> = expanded_strides(a, &out_shape);
        let b_strides: Box<[usize]> = expanded_strides(b, &out_shape);

        let data = Rc::get_mut(&mut self.data).expect("couldn't borrow mutable data from the tensor");

        LogicalIndices::new(out_shape.clone(), self_strides)
            .zip(LogicalIndices::new(out_shape.clone(), a_strides)
            .zip(LogicalIndices::new(out_shape.clone(), b_strides)))
            .for_each(|(i, (j, k))| data[i] += f(a.data[j], b.data[k]));

    }

    pub(crate) fn accumulate_3(&mut self, a: &RawTensor, b: &RawTensor, c: &RawTensor, f: impl Fn(f64, f64, f64) -> f64) {
        let out_shape: Box<[usize]> = broadcast_shape(&self.shape, &broadcast_shape(&a.shape, &broadcast_shape(&b.shape, &c.shape)));

        let self_strides: Box<[usize]> = expanded_strides(self, &out_shape);
        let a_strides: Box<[usize]> = expanded_strides(a, &out_shape);
        let b_strides: Box<[usize]> = expanded_strides(b, &out_shape);
        let c_strides: Box<[usize]> = expanded_strides(c, &out_shape);

        let data = Rc::get_mut(&mut self.data).expect("couldn't borrow mutable data from the tensor");
 
        LogicalIndices::new(out_shape.clone(), self_strides)
            .zip(LogicalIndices::new(out_shape.clone(), a_strides)
            .zip(LogicalIndices::new(out_shape.clone(), b_strides)
            .zip(LogicalIndices::new(out_shape.clone(), c_strides))))
            .for_each(|(i, (j, (k, l)))| data[i] += f(a.data[j], b.data[k], c.data[l]));
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
