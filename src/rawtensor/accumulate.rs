use super::RawTensor;
use super::iter::LogicalIndices;
use super::structure::broadcast_shape;
use super::structure::expanded_strides;
use std::rc::Rc;

// In-place accumulation used by grad/ to accumulate gradients
// Requires self to be the sole owner of its data (which is the case with gradients)

// In-place addition on self_grad of the backpropped gradient from softmax
pub fn softmax_backprop(out_grad: &[f64], out_raw: &[f64], self_grad: &mut [f64], grad_sride: usize, out_stride: usize, n: usize) {
    if out_stride == 0 {
        let mut sum: f64 = 0.0;
        (0..n).for_each(|j| {
            sum += out_grad[j * grad_sride];
        });
        (0..n).for_each(|i| {
            self_grad[i * grad_sride] += -sum * out_raw[0] * out_raw[0] + out_grad[i * grad_sride] * out_raw[0];
        });
        return;
    }

    let mut sum: f64 = 0.0;
    out_raw.iter().step_by(out_stride).take(n).enumerate().for_each(|(j, y)| {
        sum += out_grad[j * grad_sride] * y;
    });
    out_raw.iter().step_by(out_stride).take(n).enumerate().for_each(|(i, x)| {
        self_grad[i * grad_sride] += -sum * x + out_grad[i * grad_sride] * x;
    });
}

impl RawTensor {
    // In-place elementwise accumulation: self += f(a, b, c, ...) with broadcasting
    fn accumulate_n(&mut self, inputs: &[&RawTensor], f: impl Fn(&[f64]) -> f64) {
        let out_shape = inputs.iter().fold(self.shape.clone(), |acc, t| broadcast_shape(&acc, &t.shape).expect("shapes not broadcastable"));

        let self_strides = expanded_strides(self, &out_shape).expect("self strides not expandable");
        let input_strides: Box<[Box<[usize]>]> = inputs.iter()
            .map(|t| expanded_strides(t, &out_shape).expect("input strides not expandable"))
            .collect();

        let data = Rc::get_mut(&mut self.data).expect("couldn't borrow mutable data from the tensor");

        let mut iters: Box<[LogicalIndices]> = input_strides.iter()
            .map(|s| LogicalIndices::new(out_shape.clone(), s.clone()))
            .collect();

        LogicalIndices::new(out_shape, self_strides).for_each(|i| { 
            let vals: Vec<f64> = iters.iter_mut().zip(inputs.iter())
                .map(|(it, t)| t.data[it.next().expect("iterator reached end")])
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

    // In-place accumulation for softmax backpropagation
    fn accumulate_normalization_backprop(&mut self, grad: &RawTensor, out: &RawTensor, axis: usize, f: impl Fn(&[f64], &[f64], &mut [f64], usize, usize, usize)) {
        assert_eq!(self.shape, out.shape, "self gradient and out raw have different shapes");
        assert_eq!(out.shape, grad.shape, "out gradient and out raw have different shapes");
        assert!(self.is_contiguous(), "self gradient is not contiguous");
        assert!(grad.is_contiguous(), "out gradient is not contiguous");

        let n: usize = self.shape[axis];
        let out_step = out.strides[axis];
        let grad_step = grad.strides[axis];

        let grad_data: &[f64] = &grad.data;
        let out_data: &[f64] = &out.data;
        let self_data: &mut [f64] = Rc::get_mut(&mut self.data).expect("couldn't borrow mutable data from the tensor");

        if out_step == 0 {
            out.iter_indexed().zip(
                grad.iter_indexed().filter(|i| i % (n * grad_step) < grad_step))
                    .for_each(|(o, g)|
                        f(&grad_data[g..], &out_data[o..], &mut self_data[g..],
                            grad_step, out_step, n)
            );
            return;
        }

        out.iter_indexed().filter(|i| i % (n * out_step) < out_step).zip(
            grad.iter_indexed().filter(|i| i % (n * grad_step) < grad_step))
                .for_each(|(o, g)| 
                    f(&grad_data[g..], &out_data[o..], &mut self_data[g..],
                        grad_step, out_step, n)
        );
    }

    // Wrapper around accumulate_normalization_backprop for softmax
    pub(crate) fn accumulate_softmax_grad(&mut self, grad: &RawTensor, out: &RawTensor, axis: usize) {
        self.accumulate_normalization_backprop(grad, out, axis, softmax_backprop);
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
        let g = RawTensor::ones(&[1, 3]);
        let a = RawTensor::linspace(1.0, 3.0, 3).reshape(&[3, 1]);
        let b = RawTensor::linspace(1.0, 3.0, 3).reshape(&[1, 3]);
        let mut grad = RawTensor::zeros(&[3, 3]);
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

    fn approx_eq_all(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn accumulate_softmax_zero_grad() {
        let a = RawTensor::linspace(1.0, 6.0, 6).ln();
        let b = a.softmax_axis(0);
        let g = RawTensor::ones(&[6]); 
        let mut grad = RawTensor::ones(&[6]);
        grad.accumulate_softmax_grad(&g, &b, 0);
        assert!(approx_eq_all(&grad.contiguous_data(), &[1.0; 6]));
    }

    #[test]
    fn accumulate_softmax_transposed_zero_grad() {
        let a = RawTensor::linspace(1.0, 9.0, 9).ln().reshape(&[3, 3]).transpose(&[1, 0]);
        let b = a.softmax_axis(0);
        let g = RawTensor::ones(&[3, 3]); 
        let mut grad = RawTensor::ones(&[3, 3]);
        grad.accumulate_softmax_grad(&g, &b, 0);
        assert!(approx_eq_all(&grad.contiguous_data(), &[1.0; 9]));
    }

    #[test]
    fn accumulate_softmax_transposed_expanded_zero_grad() {
        let a = RawTensor::linspace(1.0, 9.0, 12).ln().reshape(&[3, 4]).transpose(&[1, 0]).expand(&[2, 4, 3]);
        let b = a.softmax_axis(1);
        let g = RawTensor::ones(&[24]).reshape(&[2, 4, 3]);
        let mut grad = RawTensor::ones(&[24]).reshape(&[2, 4, 3]);
        grad.accumulate_softmax_grad(&g, &b, 1);
        assert!(approx_eq_all(&grad.contiguous_data(), &[1.0; 24]));
    }

    #[test]
    fn accumulate_softmax_dense_grad() {
        let a = RawTensor::from_slice(&[3], &[0.0, 2.0f64.ln(), 3.0f64.ln()]);
        let b = a.softmax_axis(0);
        let g = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let mut grad = RawTensor::zeros(&[3]);
        grad.accumulate_softmax_grad(&g, &b, 0);
        assert!(approx_eq_all(&grad.contiguous_data(), &[-2.0 / 9.0, -1.0 / 9.0, 1.0 / 3.0]));
    }

    #[test]
    fn accumulate_softmax_singleton_axis() {
        let a = RawTensor::from_slice(&[3, 1], &[2.0, -5.0, 100.0]);
        let b = a.softmax_axis(1);
        let g = RawTensor::from_slice(&[3, 1], &[7.0, -2.0, 42.0]);
        let mut grad = RawTensor::from_slice(&[3, 1], &[1.0, 2.0, 3.0]);
        grad.accumulate_softmax_grad(&g, &b, 1);
        assert!(approx_eq_all(&grad.contiguous_data(), &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn accumulate_softmax_ties() {
        let a = RawTensor::from_slice(&[3], &[5.0, 5.0, 5.0]);
        let b = a.softmax_axis(0);
        let g = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let mut grad = RawTensor::zeros(&[3]);
        grad.accumulate_softmax_grad(&g, &b, 0);
        assert!(approx_eq_all(&grad.contiguous_data(), &[-1.0 / 3.0, 0.0, 1.0 / 3.0]));
    }

    #[test]
    fn accumulate_softmax_numerically_extreme() {
        let shift_up = RawTensor::from_slice(&[3], &[1e6, 1e6 + 2.0f64.ln(), 1e6 + 3.0f64.ln()]);
        let b_up = shift_up.softmax_axis(0);
        let g = RawTensor::from_slice(&[3], &[1.0, 2.0, 3.0]);
        let mut grad_up = RawTensor::zeros(&[3]);
        grad_up.accumulate_softmax_grad(&g, &b_up, 0);
        assert!(approx_eq_all(&grad_up.contiguous_data(), &[-2.0 / 9.0, -1.0 / 9.0, 1.0 / 3.0]));

        let shift_down = RawTensor::from_slice(&[3], &[-1e6, -1e6 + 2.0f64.ln(), -1e6 + 3.0f64.ln()]);
        let b_down = shift_down.softmax_axis(0);
        let mut grad_down = RawTensor::zeros(&[3]);
        grad_down.accumulate_softmax_grad(&g, &b_down, 0);
        assert!(approx_eq_all(&grad_down.contiguous_data(), &[-2.0 / 9.0, -1.0 / 9.0, 1.0 / 3.0]));
    }

    #[test]
    fn accumulate_softmax_unsqueeze_expand_transpose_batched() {
        let a = RawTensor::from_slice(&[3], &[0.0, 2.0f64.ln(), 3.0f64.ln()]).unsqueeze(0).expand(&[2, 3]).transpose(&[1, 0]);
        let b = a.softmax_axis(0);
        let g = RawTensor::from_slice(&[3, 2], &[1.0, 0.0, 2.0, 0.0, 3.0, 6.0]);
        let mut grad = RawTensor::zeros(&[3, 2]);
        grad.accumulate_softmax_grad(&g, &b, 0);
        assert!(approx_eq_all(&grad.contiguous_data(), &[-2.0 / 9.0, -0.5, -1.0 / 9.0, -1.0, 1.0 / 3.0, 1.5]));
    }

    #[test]
    fn accumulate_softmax_expanded_axis() {
        let a = RawTensor::from_slice(&[1], &[5.0]).expand(&[4]);
        let b = a.softmax_axis(0);
        let g = RawTensor::from_slice(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let mut grad = RawTensor::zeros(&[4]);
        grad.accumulate_softmax_grad(&g, &b, 0);
        assert!(approx_eq_all(&grad.contiguous_data(), &[-0.375, -0.125, 0.125, 0.375]));
    }

    #[test]
    fn accumulate_softmax_expanded_axis_batched() {
        let a = RawTensor::from_slice(&[2, 1], &[5.0, 9.0]).expand(&[2, 4]);
        let b = a.softmax_axis(1);
        let g = RawTensor::from_slice(&[2, 4], &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]);
        let mut grad = RawTensor::zeros(&[2, 4]);
        grad.accumulate_softmax_grad(&g, &b, 1);
        assert!(approx_eq_all(&grad.contiguous_data(), &[-0.375, -0.125, 0.125, 0.375, -3.75, -1.25, 1.25, 3.75]));
    }

    #[test]
    fn accumulate_softmax_transposed_expanded() {
        let a = RawTensor::from_slice(&[3, 2], &[0.0, 0.0, 2.0f64.ln(), 0.0, 3.0f64.ln(), 0.0]).transpose(&[1, 0]).expand(&[2, 2, 3]);
        let b = a.softmax_axis(2);
        let g = RawTensor::from_slice(&[2, 2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut grad = RawTensor::zeros(&[2, 2, 3]);
        grad.accumulate_softmax_grad(&g, &b, 2);
        assert!(approx_eq_all(&grad.contiguous_data(),
            &[5.0 / 36.0, -1.0 / 18.0, -1.0 / 12.0, -1.0 / 9.0, 2.0 / 9.0, -1.0 / 9.0,
              5.0 / 36.0, -1.0 / 18.0, -1.0 / 12.0, -1.0 / 9.0, 2.0 / 9.0, -1.0 / 9.0]
        ));
    }

    #[test]
    fn accumulate_softmax_transposed_squeeze() {
        let a = RawTensor::from_slice(&[3, 1, 2], &[0.0, 0.0, 2.0f64.ln(), 0.0, 3.0f64.ln(), 0.0]).transpose(&[2, 1, 0]).squeeze(1);
        let b = a.softmax_axis(1);
        let g = RawTensor::from_slice(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut grad = RawTensor::zeros(&[2, 3]);
        grad.accumulate_softmax_grad(&g, &b, 1);
        assert!(approx_eq_all(&grad.contiguous_data(),&[5.0 / 36.0, -1.0 / 18.0, -1.0 / 12.0, -1.0 / 9.0, 2.0 / 9.0, -1.0 / 9.0]));
    }

    #[test]
    fn accumulate_softmax_transposed_squeeze_expanded_accumulates() {
        let a = RawTensor::from_slice(&[3, 1, 2], &[0.0, 0.0, 2.0f64.ln(), 0.0, 3.0f64.ln(), 0.0]).transpose(&[2, 1, 0]).squeeze(1).expand(&[2, 2, 3]);
        let b = a.softmax_axis(2);
        let g = RawTensor::from_slice(&[2, 2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let mut grad = RawTensor::zeros(&[2, 2, 3]);
        grad.accumulate_softmax_grad(&g, &b, 2);
        grad.accumulate_softmax_grad(&g, &b, 2);
        assert!(approx_eq_all(&grad.contiguous_data(),
            &[5.0 / 18.0, -1.0 / 9.0, -1.0 / 6.0, -2.0 / 9.0, 4.0 / 9.0, -2.0 / 9.0,
              5.0 / 18.0, -1.0 / 9.0, -1.0 / 6.0, -2.0 / 9.0, 4.0 / 9.0, -2.0 / 9.0]
        ));
    }
}
