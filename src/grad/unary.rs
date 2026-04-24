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

// NOT IMPLEMENTED — currently passes gradient through unchanged (wrong)
pub fn sigmoid_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}

// NOT IMPLEMENTED — currently passes gradient through unchanged (wrong)
pub fn tanh_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}

// NOT IMPLEMENTED — currently passes gradient through unchanged (wrong)
pub fn abs_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}

// NOT IMPLEMENTED — currently passes gradient through unchanged (wrong)
pub fn relu_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        a_grad.accumulate_1(out_grad, |g| g);
    }
}
