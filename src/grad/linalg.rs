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

pub fn matmul_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            let b_raw_t = b.raw.transpose(&[1, 0]);
            a_grad.accumulate_1(&out_grad.matmul(&b_raw_t), |g| g);
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            let a_raw_t = a.raw.transpose(&[1, 0]);
            b_grad.accumulate_1(&a_raw_t.matmul(out_grad), |g| g);
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

    #[test]
    fn matmul_square() {
        let a = Tensor::linspace(1.0, 4.0, 4).reshape(&[2, 2]).requires_grad();
        let b = Tensor::linspace(5.0, 8.0, 4).reshape(&[2, 2]).requires_grad();
        a.matmul(&b).backward();
        assert_eq!(grad_of(&a), [11.0, 15.0, 11.0, 15.0]);
        assert_eq!(grad_of(&b), [4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn matmul_non_square() {
        let a = Tensor::linspace(1.0, 8.0, 8).reshape(&[2, 4]).requires_grad();
        let b = Tensor::linspace(1.0, 12.0, 12).reshape(&[4, 3]).requires_grad();
        a.matmul(&b).backward();
        assert_eq!(grad_of(&a), [6.0, 15.0, 24.0, 33.0, 6.0, 15.0, 24.0, 33.0]);
        assert_eq!(grad_of(&b), [6.0, 6.0, 6.0, 8.0, 8.0, 8.0, 10.0, 10.0, 10.0, 12.0, 12.0, 12.0]);
    }
}


