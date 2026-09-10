use crate::Tensor;

pub fn contiguous_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}

pub fn reshape_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let out_grad_reshaped = out_grad.reshape(a_grad.shape());
        a_grad.accumulate_1(&out_grad_reshaped, |g| g);
    }
}

pub fn transpose_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let inv_perm: Box<[usize]> = out.raw.compute_inv_perm(&a.raw);
        a_grad.accumulate_1(&out_grad.transpose(&inv_perm), |g| g);
    }
}

pub fn expand_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let out_grad_reduced = out_grad.sum_to_shape(a.shape());
        a_grad.accumulate_1(&out_grad_reduced, |g| g);
    }
}

pub fn squeeze_backprop(out: &Tensor, a: &Tensor, axis: usize) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let out_grad_unsqueezed = out_grad.unsqueeze(axis);
        a_grad.accumulate_1(&out_grad_unsqueezed, |g| g);
    }
}

pub fn unsqueeze_backprop(out: &Tensor, a: &Tensor, axis: usize) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let out_grad_squeezed = out_grad.squeeze(axis);
        a_grad.accumulate_1(&out_grad_squeezed, |g| g);
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    #[test]
    fn contiguous_grad() {
        let a = Tensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).requires_grad();
        let b = Tensor::linspace(1.0, 12.0, 12).reshape(&[2, 3, 2]).requires_grad();
        let c = a.transpose(&[1, 0]).unsqueeze(0).expand(&[2, 3, 2]).contiguous();
        (&c * &b).backward();
        assert_eq!(grad_of(&a), [8.0, 12.0, 16.0, 10.0, 14.0, 18.0]);
        assert_eq!(grad_of(&b), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn reshape_flat_to_2d() {
        let a = Tensor::linspace(1.0, 6.0, 6).requires_grad();
        let b = Tensor::from_slice(&[2, 3], &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).requires_grad();
        (&a.reshape(&[2, 3]) * &b).backward();
        assert_eq!(grad_of(&a), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(grad_of(&b), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reshape_2d_to_2d() {
        let a = Tensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).requires_grad();
        let b = Tensor::from_slice(&[3, 2], &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]).requires_grad();
        (&a.reshape(&[3, 2]) * &b).backward();
        assert_eq!(grad_of(&a), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        assert_eq!(grad_of(&b), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn transpose_2d() {
        let a = Tensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).requires_grad();
        let b = Tensor::linspace(1.0, 6.0, 6).reshape(&[3, 2]).requires_grad();
        (&a.transpose(&[1, 0]) * &b).backward();
        assert_eq!(grad_of(&a), [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
        assert_eq!(grad_of(&b), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn transpose_3d_permutation() {
        let a = Tensor::linspace(1.0, 24.0, 24).reshape(&[2, 3, 4]).requires_grad();
        let b = Tensor::linspace(1.0, 24.0, 24).reshape(&[4, 2, 3]).requires_grad();
        (&a.transpose(&[2, 0, 1]) * &b).backward();
        assert_eq!(grad_of(&a), [1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0, 21.0, 4.0, 10.0, 16.0, 22.0, 5.0, 11.0, 17.0, 23.0, 6.0, 12.0, 18.0, 24.0]);
        assert_eq!(grad_of(&b), [1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0]);
    }

    #[test]
    fn transpose_tied_shape_stride() {
        let a = Tensor::from_slice(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]).requires_grad();
        let b = Tensor::from_slice(&[1, 1, 4], &[5.0, 6.0, 7.0, 8.0]).requires_grad();
        (&a.transpose(&[1, 0, 2]) * &b).backward();
        assert_eq!(grad_of(&a), [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(grad_of(&b), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn expand_multiple_extra_leading_dims() {
        let a = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let b = Tensor::linspace(1.0, 24.0, 24).reshape(&[2, 4, 3]);
        (&a.expand(&[2, 4, 3]) * &b).backward();
        assert_eq!(grad_of(&a), [92.0, 100.0, 108.0]);
    }

    #[test]
    fn expand_multiple_size1_dims() {
        let a = Tensor::from_slice(&[1, 1, 3], &[1.0, 2.0, 3.0]).requires_grad();
        let b = Tensor::linspace(1.0, 24.0, 24).reshape(&[2, 4, 3]);
        (&a.expand(&[2, 4, 3]) * &b).backward();
        assert_eq!(grad_of(&a), [92.0, 100.0, 108.0]);
    }

    #[test]
    fn expand_extra_leading_and_size1() {
        let a = Tensor::from_slice(&[1, 3], &[1.0, 2.0, 3.0]).requires_grad();
        let b = Tensor::linspace(1.0, 24.0, 24).reshape(&[2, 4, 3]);
        (&a.expand(&[2, 4, 3]) * &b).backward();
        assert_eq!(grad_of(&a), [92.0, 100.0, 108.0]);
    }

    #[test]
    fn squeeze_axis_mid() {
        let a = Tensor::linspace(1.0, 6.0, 6).reshape(&[2, 1, 3]).requires_grad();
        let b = Tensor::from_slice(&[2, 3], &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).requires_grad();
        (&a.squeeze(1) * &b).backward();
        assert_eq!(grad_of(&a), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(grad_of(&b), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn unsqueeze_mid() {
        let a = Tensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).requires_grad();
        let b = Tensor::from_slice(&[2, 1, 3], &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]).requires_grad();
        (&a.unsqueeze(1) * &b).backward();
        assert_eq!(grad_of(&a), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(grad_of(&b), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}


