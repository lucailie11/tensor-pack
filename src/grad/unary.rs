use crate::Tensor;

pub fn exp_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &out.raw, |g, e| g * e);
    }
}

pub fn ln_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &a.raw, |g, a| g / a);
    }
}

pub fn sqrt_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &a.raw, |g, a| g / (2.0 * a.sqrt()));
    }
}

pub fn sigmoid_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &out.raw, |g, o| g * o * (1.0 - o));
    }
}

pub fn tanh_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &out.raw, |g, a| g * (1.0 - a * a));
    }
}

pub fn abs_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &a.raw, |g, a| g * (if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 }));
    }
}

pub fn relu_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &a.raw, |g, a| g * (if a > 0.0 { 1.0 } else { 0.0 }));
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    #[test]
    fn unary_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        x.ln().backward();
        assert_eq!(grad_of(&x), [1.0, 0.5, 0.25]);

        let x = Tensor::from_slice(&[3], &[1.0, 4.0, 16.0]).requires_grad();
        x.sqrt().backward();
        assert_eq!(grad_of(&x), [0.5, 0.25, 0.125]);

        let x = Tensor::from_slice(&[3], &[-2.0, 0.0, 3.0]).requires_grad();
        x.abs().backward();
        assert_eq!(grad_of(&x), [-1.0, 0.0, 1.0]);

        let x = Tensor::from_slice(&[4], &[-2.0, 0.0, 1.0, 3.0]).requires_grad();
        x.relu().backward();
        assert_eq!(grad_of(&x), [0.0, 0.0, 1.0, 1.0]);

        let x = Tensor::from_slice(&[3], &[0.0, 1.0, 2.0]).requires_grad();
        x.exp().backward();
        assert_eq!(grad_of(&x), [1.0, 1.0f64.exp(), 2.0f64.exp()]);

        let x = Tensor::from_slice(&[2], &[0.0, 1.0]).requires_grad();
        x.sigmoid().backward();
        let s1 = 1.0 / ((-1.0f64).exp() + 1.0);
        assert_eq!(grad_of(&x), [0.25, s1 * (1.0 - s1)]);

        let x = Tensor::from_slice(&[2], &[0.0, 1.0]).requires_grad();
        x.tanh().backward();
        let t1 = 1.0f64.tanh();
        assert_eq!(grad_of(&x), &[1.0, 1.0 - t1 * t1]);
    }

    #[test]
    fn unary_chain_backward() {
        let x = Tensor::from_slice(&[3], &[-2.0, 1.0, -3.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[-1.0, 2.0, 3.0]).requires_grad();
        let a = x.abs();
        let b = y.relu();
        (&a * &b).backward();
        assert_eq!(grad_of(&x), [0.0, 2.0, -3.0]);
        assert_eq!(grad_of(&y), [0.0, 1.0, 3.0]);
    }


}
