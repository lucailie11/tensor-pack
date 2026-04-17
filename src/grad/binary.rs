use crate::Tensor;

pub(super) fn add_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            *a_grad += out_grad;
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            *b_grad += out_grad;
        }
    }
}

pub(super) fn sub_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            *a_grad += out_grad;
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            *b_grad -= out_grad;
        }
    }
}
pub(super) fn mul_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            *a_grad += &(out_grad * &b.raw);
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            *b_grad += &(out_grad * &a.raw);
        }
    }
}

pub(super) fn div_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() {
        if let Some(a_grad) = a.grad.borrow_mut().as_mut() {
            *a_grad += &(out_grad / &b.raw);
        }
        if let Some(b_grad) = b.grad.borrow_mut().as_mut() {
            *b_grad -= &(out_grad * &(&a.raw / &(&b.raw * &b.raw)));
        }
    }
}

