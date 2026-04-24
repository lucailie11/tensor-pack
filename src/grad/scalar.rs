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
