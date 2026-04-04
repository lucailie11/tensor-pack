use std::fmt;

// Core RawTensor type: a pair of shape and data (stored in row-major order).
// Also defines the standard constructors.

#[derive(Debug)]
pub struct RawTensor {
    pub(super) shape: Box<[usize]>,
    pub(super) data: Box<[f64]>,
}

impl RawTensor {
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn data(&self) -> &[f64]    { &self.data  }
}

// Pretty-prints the RawTensor. 2D tensors are shown row-by-row; all other ranks print the flat data slice.
impl fmt::Display for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "RawTensor {{")?;
        writeln!(f, "    shape: {:?}", self.shape)?;
        writeln!(f, "    data: ")?;
        if self.shape.len() == 2 {
            let cols = self.shape[1];
            for row in self.data.chunks(cols) {
                let formatted: Vec<String> = row.iter().map(|x| format!("{:.4}", x)).collect();
                writeln!(f, "           [{}]", formatted.join(", "))?;
            }
        } else {
            write!(f, "{:.4?}", self.data)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}
