use crate::Tensor;

pub fn sum_backprop(out: &Tensor, a: &Tensor, axis: usize) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let out_grad_unsqueezed = out_grad.unsqueeze(axis);
        a_grad.accumulate_1(&out_grad_unsqueezed, |g| g);
    }
}

pub fn mean_backprop(out: &Tensor, a: &Tensor, axis: usize) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let out_grad_unsqueezed = out_grad.unsqueeze(axis);
        a_grad.accumulate_1(&out_grad_unsqueezed, |g| g / a.raw.shape()[axis] as f64);
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    #[test]
    fn reductions_backward() {
        let x = Tensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).requires_grad();
        let y = x.sum_axis(1);
        y.backward();
        assert_eq!(grad_of(&x), [1.0; 6]);
        
        let x = Tensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).requires_grad();
        let y = x.mean_axis(1);
        y.backward();
        assert_eq!(grad_of(&x), [1.0 / 3.0; 6]);
    }

    // TODO
}


