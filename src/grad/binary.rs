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
