use crate::Tensor;

// TODO
pub fn softmax_backprop(out: &Tensor, a: &Tensor, axis: usize) {
    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    fn grad_of(t: &Tensor) -> Vec<f64> {
        t.grad.borrow().as_ref().expect("no grad").contiguous_data().to_vec()
    }

    #[test]
    // TODO
    fn normalizations_backward() {
        let x = Tensor::linspace(1.0, 6.0, 6).reshape(&[2, 3]).requires_grad();
        let y = x.softmax(1);
        y.backward();
        // assert_eq!(grad_of(&x), [1.0; 6]);
    }
    // TODO
}


