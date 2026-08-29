use crate::Tensor;

pub fn softmax_backprop(out: &Tensor, a: &Tensor, axis: usize) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_softmax_backprop(out_grad, &out.raw, axis);
    }
 }

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    fn approx_eq_all(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn softmax_backward_uniform_grad_is_zero() {
        let a = Tensor::from_slice(&[3], &[0.0, 2.0f64.ln(), 3.0f64.ln()]).requires_grad();
        a.softmax_axis(0).backward();
        assert!(approx_eq_all(&grad_of(&a), &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn softmax_backward_weighted_loss() {
        let a = Tensor::from_slice(&[3], &[0.0, 2.0f64.ln(), 3.0f64.ln()]).requires_grad();
        let w = Tensor::from_slice(&[3], &[1.0, 0.0, 0.0]);
        let loss = &a.softmax_axis(0) * &w;
        loss.sum_axis(0).backward();
        assert!(approx_eq_all(&grad_of(&a), &[5.0 / 36.0, -1.0 / 18.0, -1.0 / 12.0]));
    }

    #[test]
    fn softmax_backward_respects_axis() {
        let a = Tensor::from_slice(&[2, 3], &[0.0, 2.0f64.ln(), 3.0f64.ln(), 0.0, 0.0, 0.0]).requires_grad();
        let w = Tensor::from_slice(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let loss = &a.softmax_axis(1) * &w;
        loss.sum_axis(1).backward();
        assert!(approx_eq_all(&grad_of(&a), &[5.0 / 36.0, -1.0 / 18.0, -1.0 / 12.0, -1.0 / 9.0, 2.0 / 9.0, -1.0 / 9.0]));
    }
}
