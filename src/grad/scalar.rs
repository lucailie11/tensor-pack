use crate::Tensor;

pub fn add_scalar_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}

pub fn sub_scalar_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| -g);
    }
}

pub fn mul_scalar_backprop(out: &Tensor, a: &Tensor, scalar: f64) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g * scalar);
    }
}

pub fn div_scalar_backprop(out: &Tensor, a: &Tensor, scalar: f64) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &a.raw, |g, a| -g * scalar / (a * a));
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    #[test]
    fn scalar_backward() {
        let x = Tensor::linspace(1.0, 3.0, 3).requires_grad();
        (&x + 5.0).backward();
        assert_eq!(grad_of(&x), [1.0, 1.0, 1.0]);

        let x = Tensor::linspace(1.0, 3.0, 3).requires_grad();
        (5.0_f64 - &x).backward();
        assert_eq!(grad_of(&x), [-1.0, -1.0, -1.0]);

        let x = Tensor::linspace(1.0, 3.0, 3).requires_grad();
        (&x * 4.0).backward();
        assert_eq!(grad_of(&x), [4.0, 4.0, 4.0]);

        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        (8.0_f64 / &x).backward();
        assert_eq!(grad_of(&x), [-8.0, -2.0, -0.5]);
    }

    #[test]
    fn scalar_chain_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[2.0, 1.0, 2.0]).requires_grad();
        let a = &x * &y;
        (&a * 3.0).backward();
        assert_eq!(grad_of(&x), [6.0, 3.0, 6.0]);
        assert_eq!(grad_of(&y), [3.0, 6.0, 12.0]);
    }


}
