use crate::{Tensor, rawtensor::RawTensor, tensor::core::TensorInner};
use std::{cell::RefMut};

pub fn add_tensor_backprop(output: &Tensor, inputs: &[Tensor]) {
    assert_eq!(inputs.len(), 2, "Addition is a binary operation");
    assert!(output.borrow().grad.is_some(), "The output tensor should already have a grad");

    for input in inputs.iter() {
        let mut tensor: RefMut<TensorInner> = input.borrow_mut();
        if tensor.grad.is_none() {
            tensor.grad = Some(RawTensor::zeros(tensor.raw.shape()));
        }
    }


    if let Some(out_grad) = output.borrow().grad.as_ref() && let Some(a_grad) = inputs[0].borrow_mut().grad.as_mut() {
        *a_grad += out_grad;
    }
 
    if let Some(out_grad) = output.borrow().grad.as_ref() && let Some(b_grad) = inputs[1].borrow_mut().grad.as_mut() {
        *b_grad += out_grad;
    }
}

pub fn sub_tensor_backprop(output: &Tensor, inputs: &[Tensor]) {
    assert_eq!(inputs.len(), 2, "Substraction is a binary operation");
    assert!(output.borrow().grad.is_some(), "The output tensor should already have a grad");

    for input in inputs.iter() {
        let mut tensor: RefMut<TensorInner> = input.borrow_mut();
        if tensor.grad.is_none() {
            tensor.grad = Some(RawTensor::zeros(tensor.raw.shape()));
        }
    }

    if let Some(out_grad) = output.borrow().grad.as_ref() && let Some(a_grad) = inputs[0].borrow_mut().grad.as_mut() {
        *a_grad += out_grad;
    }
 
    if let Some(out_grad) = output.borrow().grad.as_ref() && let Some(b_grad) = inputs[1].borrow_mut().grad.as_mut() {
        *b_grad -= out_grad;
    }
}

pub fn mul_tensor_backprop(output: &Tensor, inputs: &[Tensor]) {
    assert_eq!(inputs.len(), 2, "Multiplication is a binary operation");
    assert!(output.borrow().grad.is_some(), "The output tensor should already have a grad");

    for input in inputs.iter() {
        let mut tensor: RefMut<TensorInner> = input.borrow_mut();
        if tensor.grad.is_none() {
            tensor.grad = Some(RawTensor::zeros(tensor.raw.shape()));
        }
    }

    if let Some(out_grad) = output.borrow().grad.as_ref() {
        let b_raw: &RawTensor = &inputs[1].borrow().raw;
        println!("aaa");
        if let Some(a_grad) = inputs[0].borrow_mut().grad.as_mut() {
            *a_grad += &(out_grad * b_raw);
        }
    }
 
    if let Some(out_grad) = output.borrow().grad.as_ref() {
        let a_raw: &RawTensor = &inputs[0].borrow().raw;
        if let Some(b_grad) = inputs[1].borrow_mut().grad.as_mut() {
            *b_grad += &(out_grad * a_raw);
        }
    }
}

pub fn div_tensor_backprop(output: &Tensor, inputs: &[Tensor]) {
    assert_eq!(inputs.len(), 2, "Multiplication is a binary operation");
    assert!(output.borrow().grad.is_some(), "The output tensor should already have a grad");

    for input in inputs.iter() {
        let mut tensor: RefMut<TensorInner> = input.borrow_mut();
        if tensor.grad.is_none() {
            tensor.grad = Some(RawTensor::zeros(tensor.raw.shape()));
        }
    }

    if let Some(out_grad) = output.borrow().grad.as_ref() {
        let b_raw: &RawTensor = &inputs[1].borrow().raw;
        if let Some(a_grad) = inputs[0].borrow_mut().grad.as_mut() {
            *a_grad += &(out_grad / b_raw);
        }
    }
 
    if let Some(out_grad) = output.borrow().grad.as_ref() {
        let a_raw: &RawTensor = &inputs[0].borrow().raw;
        let b_raw: &RawTensor = &inputs[1].borrow().raw;
        if let Some(b_grad) = inputs[1].borrow_mut().grad.as_mut() {
            *b_grad -= &(out_grad * &(a_raw / &(b_raw * b_raw)));
        }
    }
}


