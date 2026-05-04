pub mod core;
mod binary;
mod backward;
mod scalar;
mod unary;

pub(crate) use core::BackpropOp;

#[cfg(test)]
mod tests {
    use crate::Tensor;

    const TOL: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").data().to_vec()
    }

    #[test]
    fn add_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[4.0, 5.0, 6.0]).requires_grad();
        (&x + &y).backward();
        assert!(grad_of(&x).iter().all(|&g| approx(g, 1.0)));
        assert!(grad_of(&y).iter().all(|&g| approx(g, 1.0)));
    }

    #[test]
    fn sub_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[4.0, 5.0, 6.0]).requires_grad();
        (&x - &y).backward();
        assert!(grad_of(&x).iter().all(|&g| approx(g,  1.0)));
        assert!(grad_of(&y).iter().all(|&g| approx(g, -1.0)));
    }

    #[test]
    fn mul_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[4.0, 5.0, 6.0]).requires_grad();
        (&x * &y).backward();
        assert_eq!(grad_of(&x), vec![4.0, 5.0, 6.0]);
        assert_eq!(grad_of(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn div_backward() {
        let x = Tensor::from_slice(&[3], &[2.0, 4.0, 6.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        (&x / &y).backward();
        let gx = grad_of(&x);
        let gy = grad_of(&y);
        assert!(approx(gx[0], 1.0) && approx(gx[1], 0.5) && approx(gx[2], 1.0 / 3.0));
        assert!(approx(gy[0], -2.0) && approx(gy[1], -1.0) && approx(gy[2], -2.0 / 3.0));
    }

    #[test]
    fn add_scalar_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        (&x + 5.0).backward();
        assert!(grad_of(&x).iter().all(|&g| approx(g, 1.0)));
    }

    #[test]
    fn sub_scalar_rhs_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        (&x - 5.0).backward();
        assert!(grad_of(&x).iter().all(|&g| approx(g, 1.0)));
    }

    #[test]
    fn sub_scalar_lhs_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        (5.0 - &x).backward();
        assert!(grad_of(&x).iter().all(|&g| approx(g, -1.0)));
    }

    #[test]
    fn mul_scalar_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        (&x * 3.0).backward();
        assert!(grad_of(&x).iter().all(|&g| approx(g, 3.0)));
    }

    #[test]
    fn div_scalar_rhs_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        (&x / 4.0).backward();
        assert!(grad_of(&x).iter().all(|&g| approx(g, 0.25)));
    }

    #[test]
    fn div_scalar_lhs_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        (4.0 / &x).backward();
        let gx = grad_of(&x);
        assert!(approx(gx[0], -4.0) && approx(gx[1], -1.0) && approx(gx[2], -0.25));
    }

    #[test]
    fn exp_backward() {
        let x = Tensor::from_slice(&[3], &[0.0, 1.0, 2.0]).requires_grad();
        x.exp().backward();
        let gx = grad_of(&x);
        assert!(approx(gx[0], 1.0));
        assert!(approx(gx[1], 1.0_f64.exp()));
        assert!(approx(gx[2], 2.0_f64.exp()));
    }

    #[test]
    fn ln_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 4.0]).requires_grad();
        x.ln().backward();
        let gx = grad_of(&x);
        assert!(approx(gx[0], 1.0) && approx(gx[1], 0.5) && approx(gx[2], 0.25));
    }

    #[test]
    fn sqrt_backward() {
        let x = Tensor::from_slice(&[3], &[1.0, 4.0, 9.0]).requires_grad();
        x.sqrt().backward();
        let gx = grad_of(&x);
        assert!(approx(gx[0], 0.5) && approx(gx[1], 0.25) && approx(gx[2], 1.0 / 6.0));
    }

    #[test]
    fn sigmoid_backward() {
        let x = Tensor::from_slice(&[3], &[0.0, 1.0, -1.0]).requires_grad();
        x.sigmoid().backward();
        let s = |v: f64| 1.0 / (1.0 + (-v).exp());
        let gx = grad_of(&x);
        for (g, &xi) in gx.iter().zip(&[0.0, 1.0, -1.0]) {
            assert!(approx(*g, s(xi) * (1.0 - s(xi))));
        }
    }

    #[test]
    #[ignore = "tanh backprop not yet implemented"]
    fn tanh_backward() {
        let x = Tensor::from_slice(&[3], &[0.0, 1.0, -1.0]).requires_grad();
        x.tanh().backward();
        let gx = grad_of(&x);
        for (g, &xi) in gx.iter().zip(&[0.0, 1.0, -1.0_f64]) {
            assert!(approx(*g, 1.0 - xi.tanh().powi(2)));
        }
    }

    #[test]
    fn abs_backward() {
        let x = Tensor::from_slice(&[3], &[-2.0, 1.0, 3.0]).requires_grad();
        x.abs().backward();
        let gx = grad_of(&x);
        assert!(approx(gx[0], -1.0) && approx(gx[1], 1.0) && approx(gx[2], 1.0));
    }

    #[test]
    fn relu_backward() {
        let x = Tensor::from_slice(&[4], &[-2.0, -0.5, 1.0, 3.0]).requires_grad();
        x.relu().backward();
        let gx = grad_of(&x);
        assert!(approx(gx[0], 0.0) && approx(gx[1], 0.0) && approx(gx[2], 1.0) && approx(gx[3], 1.0));
    }

    #[test]
    fn chain_mul_add() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[2.0, 3.0, 4.0]).requires_grad();
        (&(&x * &y) + &x).backward();
        assert_eq!(grad_of(&x), vec![3.0, 4.0, 5.0]);
        assert_eq!(grad_of(&y), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn chain_exp_mul() {
        let x = Tensor::from_slice(&[2], &[0.0, 1.0]).requires_grad();
        let y = Tensor::from_slice(&[2], &[2.0, 3.0]).requires_grad();
        (&x.exp() * &y).backward();
        let gx = grad_of(&x);
        let gy = grad_of(&y);
        assert!(approx(gx[0], 2.0) && approx(gx[1], 3.0 * 1.0_f64.exp()));
        assert!(approx(gy[0], 1.0) && approx(gy[1], 1.0_f64.exp()));
    }

    #[test]
    fn chain_div_ln() {
        let x = Tensor::from_slice(&[2], &[2.0, 4.0]).requires_grad();
        let y = Tensor::from_slice(&[2], &[2.0, 2.0]).requires_grad();
        (&x.ln() / &y).backward();
        let gx = grad_of(&x);
        let gy = grad_of(&y);
        assert!(approx(gx[0], 0.25) && approx(gx[1], 0.125));
        assert!(approx(gy[0], -2.0_f64.ln() / 4.0) && approx(gy[1], -4.0_f64.ln() / 4.0));
    }

    #[test]
    fn grad_accumulates_when_used_twice() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        (&x + &x).backward();
        assert!(grad_of(&x).iter().all(|&g| approx(g, 2.0)));
    }

    #[test]
    fn no_grad_tensor_gets_no_gradient() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        let y = Tensor::from_slice(&[3], &[4.0, 5.0, 6.0]);
        (&x + &y).backward();
        assert!(y.grad.borrow().is_none());
    }

    #[test]
    fn same_var_twice() {
        let x = Tensor::from_slice(&[3], &[1.0, 2.0, 3.0]).requires_grad();
        (&x * &x).backward();
        (&x / &x).backward();
    }
}

