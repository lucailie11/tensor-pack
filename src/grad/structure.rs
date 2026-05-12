use crate::Tensor;

// TODO
pub fn reshape_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
    }
}

// TODO
pub fn transpose_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
    }
}

// TODO
pub fn expand_backprop(out: &Tensor, a: &Tensor) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
    }
}

pub fn squeeze_backprop(out: &Tensor, a: &Tensor, axis: usize) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let out_grad_unsqueezed = out_grad.unsqueeze_axis(axis);
        a_grad.accumulate_1(&out_grad_unsqueezed, |g| g);
    }
}

pub fn unsqueeze_backprop(out: &Tensor, a: &Tensor, axis: usize) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        let out_grad_squeezed = out_grad.squeeze_axis(axis);
        a_grad.accumulate_1(&out_grad_squeezed, |g| g);
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    // TODO
}


