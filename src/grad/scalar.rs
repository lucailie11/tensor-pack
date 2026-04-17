use crate::Tensor;

pub(super) fn add_scalar_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        *a_grad += out_grad;
    }
}

pub(super) fn sub_scalar_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        *a_grad -= out_grad;
    }
}
pub(super) fn mul_scalar_backprop(out: &Tensor, a: &Tensor, scalar: f64) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        *a_grad += &(out_grad * scalar);
    }
}

pub(super) fn div_scalar_backprop(out: &Tensor, a: &Tensor, scalar: f64) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        *a_grad -= &(out_grad * &(scalar / &(&a.raw * &a.raw)));
    }
}
