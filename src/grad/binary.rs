use crate::{Tensor, rawtensor::RawTensor};

pub fn init_input_grad(input: &Tensor) {
    if input.grad.borrow().is_none() {
        *input.grad.borrow_mut() = Some(RawTensor::zeros(input.shape()));
    }
}

pub fn add_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    assert!(out.grad.borrow().is_some(), "The output tensor should already have a grad");

    init_input_grad(a);
    init_input_grad(b);

    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        *a_grad += out_grad;
    }

    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(b_grad) = b.grad.borrow_mut().as_mut() {
        *b_grad += out_grad;
    }
}

pub fn sub_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    assert!(out.grad.borrow().is_some(), "The output tensor should already have a grad");

    init_input_grad(a);
    init_input_grad(b);

    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        *a_grad += out_grad;
    }

    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(b_grad) = b.grad.borrow_mut().as_mut() {
        *b_grad -= out_grad;
    }
}
pub fn mul_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    assert!(out.grad.borrow().is_some(), "The output tensor should already have a grad");

    init_input_grad(a);
    init_input_grad(b);

    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        *a_grad += &(out_grad * &b.raw);
    }

    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(b_grad) = b.grad.borrow_mut().as_mut() {
        *b_grad += &(out_grad * &a.raw);
    }
}

pub fn div_tensor_backprop(out: &Tensor, a: &Tensor, b: &Tensor) {
    assert!(out.grad.borrow().is_some(), "The output tensor should already have a grad");

    init_input_grad(a);
    init_input_grad(b);

    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(a_grad) = a.grad.borrow_mut().as_mut() {
        *a_grad += &(out_grad / &b.raw);
    }

    if let Some(out_grad) = out.grad.borrow().as_ref() && let Some(b_grad) = b.grad.borrow_mut().as_mut() {
        *b_grad -= &(out_grad * &(&a.raw / &(&b.raw * &b.raw)));
    }
}
// pub fn sub_tensor_backprop<'a>(output: &Tensor, inputs: &[&'a Tensor<'a>]) {
//     assert_eq!(inputs.len(), 2, "Substraction is a binary operation");
//     assert!(output.grad.borrow().is_some(), "The output tensor should already have a grad");
//
//     init_input_grads(inputs);
//
//     if let Some(out_grad) = output.grad.borrow().as_ref() && let Some(a_grad) = inputs[0].grad.borrow_mut().as_mut() {
//         *a_grad += out_grad;
//     }
//
//     if let Some(out_grad) = output.grad.borrow().as_ref() && let Some(b_grad) = inputs[1].grad.borrow_mut().as_mut() {
//         *b_grad -= out_grad;
//     }
// }
//
// pub fn mul_tensor_backprop<'a>(output: &Tensor, inputs: &[&'a Tensor<'a>]) {
//     assert_eq!(inputs.len(), 2, "Multiplication is a binary operation");
//     assert!(output.grad.borrow().is_some(), "The output tensor should already have a grad");
//
//     init_input_grads(inputs);
//
//     let a_raw: &RawTensor = &inputs[0].raw;
//     let b_raw: &RawTensor = &inputs[1].raw;
//
//     if let Some(out_grad) = output.grad.borrow().as_ref() && let Some(a_grad) = inputs[0].grad.borrow_mut().as_mut() {
//         *a_grad += &(out_grad * a_raw);
//     }
//
//     if let Some(out_grad) = output.grad.borrow().as_ref() && let Some(b_grad) = inputs[1].grad.borrow_mut().as_mut() {
//         *b_grad += &(out_grad * b_raw);
//     }
// }
//
// pub fn div_tensor_backprop<'a>(output: &Tensor, inputs: &[&'a Tensor<'a>]) {
//     assert_eq!(inputs.len(), 2, "Division is a binary operation");
//     assert!(output.grad.borrow().is_some(), "The output tensor should already have a grad");
//
//     init_input_grads(inputs);
//
//     let a_raw: &RawTensor = &inputs[0].raw;
//     let b_raw: &RawTensor = &inputs[1].raw;
//
//     if let Some(out_grad) = output.grad.borrow().as_ref() && let Some(a_grad) = inputs[0].grad.borrow_mut().as_mut() {
//         *a_grad += &(out_grad / a_raw);
//     }
//
//     if let Some(out_grad) = output.grad.borrow().as_ref() && let Some(b_grad) = inputs[1].grad.borrow_mut().as_mut() {
//         *b_grad -= &(out_grad * &(a_raw / &(b_raw * b_raw)));
//     }
// }
//
