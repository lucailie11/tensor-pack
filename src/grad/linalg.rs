use crate::Tensor;

pub fn dot_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() { 
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            a_grad.accumulate_2(out_grad, &b.raw, |g, b| g * b);
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            b_grad.accumulate_2(out_grad, &a.raw, |g, a| g * a);
        }
    }
}

// TODO
pub fn matmul_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
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
    fn dot_basic() {
        let x = Tensor::linspace(1.0, 5.0, 5).requires_grad();
        let y = Tensor::linspace(2.0, 10.0, 5).requires_grad();
        let z = x.dot(&y);
        z.backward();
        assert_eq!(grad_of(&x), [2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(grad_of(&y), [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn dot_self() {
        let x = Tensor::linspace(1.0, 5.0, 5).requires_grad();
        let y = x.dot(&x);
        y.backward();
        assert_eq!(grad_of(&x), [2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    // TODO
}


