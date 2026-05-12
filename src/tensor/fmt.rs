use super::Tensor;

use core::fmt;

// Brute-force: shape, strides, raw backing data, requires_grad, raw grad data
impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vals: Vec<String> = self.raw.data().iter().map(|x| format!("{:.4}", x)).collect();
        write!(f, "Tensor {{ shape: {:?}, requires_grad: {}, data: [{}]",
            self.shape(), self.requires_grad, vals.join(", "))?;
        if let Some(grad) = self.grad.borrow().as_ref() {
            let gvals: Vec<String> = grad.data().iter().map(|x| format!("{:.4}", x)).collect();
            write!(f, ", grad: [{}]", gvals.join(", "))?;
        }
        write!(f, " }}")
    }
}

// Logical order: shape, requires_grad, data, and grad if present. 2D tensors are shown as a matrix.
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Tensor {{")?;
        writeln!(f, "    shape:         {:?}", self.shape())?;
        writeln!(f, "    requires_grad: {}", self.requires_grad)?;
        writeln!(f, "    data:")?;
        self.raw.fmt_data(f, "        ")?;
        if let Some(grad) = self.grad.borrow().as_ref() {
            writeln!(f, "    grad:")?;
            grad.fmt_data(f, "        ")?;
        }
        write!(f, "}}")
    }
}
