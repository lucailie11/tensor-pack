use crate::Tensor;

pub fn add_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            a_grad.accumulate_1(out_grad, |g| g);
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            b_grad.accumulate_1(out_grad, |g| g);
        }
    }
}

pub fn sub_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            a_grad.accumulate_1(out_grad, |g| g);
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            b_grad.accumulate_1(out_grad, |g| -g);
        }
    }
}

pub fn mul_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            a_grad.accumulate_2(out_grad, &b.raw, |g, b| g * b);
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            b_grad.accumulate_2(out_grad, &a.raw, |g, a| g * a);
        }
    }
}

pub fn div_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            a_grad.accumulate_2(out_grad, &b.raw, |g, b| g / b);
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            b_grad.accumulate_3(out_grad, &a.raw, &b.raw, |g, a, b| g * (-a) / (b * b));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    #[test]
    fn binary_backward() {
        let x = Tensor::linspace(1.0, 3.0, 3).requires_grad();
        let y = Tensor::from_slice(&[3], &[4.0, 5.0, 6.0]).requires_grad();
        (&x + &y).backward();
        assert_eq!(grad_of(&x), [1.0, 1.0, 1.0]);
        assert_eq!(grad_of(&y), [1.0, 1.0, 1.0]);

        let x = Tensor::linspace(1.0, 3.0, 3).requires_grad();
        let y = Tensor::from_slice(&[3], &[4.0, 5.0, 6.0]).requires_grad();
        (&x - &y).backward();
        assert_eq!(grad_of(&x), [1.0, 1.0, 1.0]);
        assert_eq!(grad_of(&y), [-1.0, -1.0, -1.0]);

        let x = Tensor::linspace(1.0, 3.0, 3).requires_grad();
        let y = Tensor::from_slice(&[3], &[4.0, 5.0, 6.0]).requires_grad();
        (&x * &y).backward();
        assert_eq!(grad_of(&x), [4.0, 5.0, 6.0]);
        assert_eq!(grad_of(&y), [1.0, 2.0, 3.0]);

        let x = Tensor::from_slice(&[3], &[2.0, 4.0, 6.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        (&x / &y).backward();
        assert_eq!(grad_of(&x), [1.0, 0.5, 0.25]);
        assert_eq!(grad_of(&y), [-2.0, -1.0, -0.375]);
    }

    #[test]
    fn same_tensor_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        (&x + &x).backward();
        assert_eq!(grad_of(&x), [2.0, 2.0, 2.0]);

        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        (&x - &x).backward();
        assert_eq!(grad_of(&x), [0.0, 0.0, 0.0]);

        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        (&x * &x).backward();
        assert_eq!(grad_of(&x), [2.0, 4.0, 8.0]);

        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        (&x / &x).backward();
        assert_eq!(grad_of(&x), [0.0, 0.0, 0.0]);
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
