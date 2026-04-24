use crate::Tensor;

pub fn exp_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_2(out_grad, &a.raw, |g, a| g * a.exp());
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

// TODO
pub fn sigmoid_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}

// TODO
pub fn tanh_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}

// TODO
pub fn abs_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}

// TODO
pub fn relu_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}
