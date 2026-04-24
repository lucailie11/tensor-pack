use super::Tensor;
use core::fmt;

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor {{ data: {:?}", self.raw)?;
        write!(f, ", requires_grad: {}", self.requires_grad)?;
        if let Some(grad) = self.grad.borrow().as_ref() {
            write!(f, ", grad: {:?}", grad)?;
        }
        write!(f, " }}")
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use crate::rawtensor::fmt_matrix;
        writeln!(f, "Tensor {{")?;
        writeln!(f, "    shape:         {:?}", self.raw.shape())?;
        writeln!(f, "    requires_grad: {}", self.requires_grad)?;
        writeln!(f, "    op: {:?}", self.op)?;
        if self.raw.shape().len() == 2 {
            writeln!(f, "    data:")?;
            fmt_matrix(f, self.raw.data(), self.raw.shape()[1], "        ")?;
        } else {
            let vals: Vec<String> = self.raw.data().iter().map(|x| format!("{:.4}", x)).collect();
            writeln!(f, "    data:          [{}]", vals.join(", "))?;
        }
        if let Some(grad) = self.grad.borrow().as_ref() {
            if grad.shape().len() == 2 {
                writeln!(f, "    grad:")?;
                fmt_matrix(f, grad.data(), grad.shape()[1], "        ")?;
            } else {
                let vals: Vec<String> = grad.data().iter().map(|x| format!("{:.4}", x)).collect();
                writeln!(f, "    grad:          [{}]", vals.join(", "))?;
            }
        }
        write!(f, "}}")
    }
}


