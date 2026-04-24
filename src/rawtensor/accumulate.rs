use std::rc::Rc;
use super::RawTensor;
use super::iter::LogicalIndices;
use super::structure::broadcast_shape;

// In-place gradient accumulation: self += f(a, b, c, ...) with broadcasting
// Used by grad/ to accumulate gradients
// Requires self to be the sole owner of its data

impl RawTensor {
    pub(crate) fn accumulate_1(&mut self, a: &RawTensor, f: impl Fn(f64) -> f64) {
        let out_shape: Box<[usize]> = broadcast_shape(&self.shape, &a.shape);

        let self_strides: Box<[usize]> = self.expand_strides(&out_shape);
        let a_strides: Box<[usize]> = a.expand_strides(&out_shape);

        let data = Rc::get_mut(&mut self.data).expect("couldn't borrow mutable data from the tensor");

        LogicalIndices::new(out_shape.clone(), self_strides)
            .zip(LogicalIndices::new(out_shape.clone(), a_strides))
            .for_each(|(i, j)| data[i] += f(a.data[j]));
    }

    pub(crate) fn accumulate_2(&mut self, a: &RawTensor, b: &RawTensor, f: impl Fn(f64, f64) -> f64) {
        let out_shape: Box<[usize]> = broadcast_shape(&self.shape, &broadcast_shape(&a.shape, &b.shape));

        let self_strides: Box<[usize]> = self.expand_strides(&out_shape);
        let a_strides: Box<[usize]> = a.expand_strides(&out_shape);
        let b_strides: Box<[usize]> = b.expand_strides(&out_shape);

        let data = Rc::get_mut(&mut self.data).expect("couldn't borrow mutable data from the tensor");

        LogicalIndices::new(out_shape.clone(), self_strides)
            .zip(LogicalIndices::new(out_shape.clone(), a_strides)
            .zip(LogicalIndices::new(out_shape.clone(), b_strides)))
            .for_each(|(i, (j, k))| data[i] += f(a.data[j], b.data[k]));

    }

    pub(crate) fn accumulate_3(&mut self, a: &RawTensor, b: &RawTensor, c: &RawTensor, f: impl Fn(f64, f64, f64) -> f64) {
        let out_shape: Box<[usize]> = broadcast_shape(&self.shape, &broadcast_shape(&a.shape, &broadcast_shape(&b.shape, &c.shape)));

        let self_strides: Box<[usize]> = self.expand_strides(&out_shape);
        let a_strides: Box<[usize]> = a.expand_strides(&out_shape);
        let b_strides: Box<[usize]> = b.expand_strides(&out_shape);
        let c_strides: Box<[usize]> = c.expand_strides(&out_shape);

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
    fn accumulate_1_adds_transformed_grad() {
        let mut grad = RawTensor::zeros(&[3]);
        let g = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        grad.accumulate_1(&g, |gv| gv * 2.0);
        assert_eq!(grad.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn accumulate_2_div_backward_a() {
        // forward: out = a / b, backward for a: out.grad / b
        let mut grad = RawTensor::zeros(&[3]);
        let g = RawTensor::from_slice(&[3], &[1.0, 1.0, 1.0]);
        let b = RawTensor::from_slice(&[3], &[2.0, 4.0, 5.0]);
        grad.accumulate_2(&g, &b, |gv, bv| gv / bv);
        assert_eq!(grad.data(), &[0.5, 0.25, 0.2]);
    }

    #[test]
    fn accumulate_3_div_backward_b() {
        // forward: out = a / b, backward for b: out.grad * (-a) / (b * b)
        let mut grad = RawTensor::zeros(&[3]);
        let g = RawTensor::from_slice(&[3], &[1.0, 1.0, 1.0]);
        let a = RawTensor::from_slice(&[3], &[6.0, 4.0, 9.0]);
        let b = RawTensor::from_slice(&[3], &[2.0, 2.0, 3.0]);
        grad.accumulate_3(&g, &a, &b, |gv, av, bv| gv * (-av) / (bv * bv));
        assert_eq!(grad.data(), &[-1.5, -1.0, -1.0]);
    }

    #[test]
    fn accumulate_is_additive() {
        // calling twice accumulates, doesn't overwrite
        let mut grad = RawTensor::from_slice(&[2], &[1.0, 1.0]);
        let g = RawTensor::from_slice(&[2], &[2.0, 3.0]);
        grad.accumulate_1(&g, |gv| gv);
        grad.accumulate_1(&g, |gv| gv);
        assert_eq!(grad.data(), &[5.0, 7.0]);
    }
}
