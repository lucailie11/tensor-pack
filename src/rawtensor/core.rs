use std::fmt;

// Core RawTensor type: a pair of shape and data (stored in row-major order).

#[derive(Clone, PartialEq)] 
pub struct RawTensor {
    pub(super) shape: Box<[usize]>,
    pub(super) data: Box<[f64]>,
}

impl RawTensor {
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn data(&self) -> &[f64]    { &self.data  }

    // Changes the shape without moving data
    // Panics if the length changes
    pub fn reshape(&mut self, new_shape: &[usize]) {
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.data.len(),
            "New shape doesn't match old data length"
        );
        self.shape = Box::from(new_shape);
    }

}

// Writes a 2D slice as column-aligned rows, each indented by `indent`.
pub(crate) fn fmt_matrix(f: &mut fmt::Formatter, data: &[f64], cols: usize, indent: &str) -> fmt::Result {
    let formatted: Vec<Vec<String>> = data.chunks(cols)
        .map(|row| row.iter().map(|x| format!("{:.4}", x)).collect())
        .collect();
    let col_widths: Vec<usize> = (0..cols)
        .map(|c| formatted.iter().map(|row| row[c].len()).max().unwrap_or(0))
        .collect();
    for row in &formatted {
        let padded: Vec<String> = row.iter().zip(&col_widths)
            .map(|(val, &w)| format!("{:>width$}", val, width = w))
            .collect();
        writeln!(f, "{}[{}]", indent, padded.join(", "))?;
    }
    Ok(())
}

impl fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RawTensor {{ shape: {:?}, data: ", self.shape)?;
        let vals: Vec<String> = self.data.iter().map(|x| format!("{:.4}", x)).collect();
        write!(f, "[{}]", vals.join(", "))?;
        write!(f, " }}")
    }
}

impl fmt::Display for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "RawTensor {{")?;
        writeln!(f, "    shape: {:?}", self.shape)?;
        if self.shape.len() == 2 {
            writeln!(f, "    data:")?;
            fmt_matrix(f, &self.data, self.shape[1], "        ")?;
        } else {
            let vals: Vec<String> = self.data.iter().map(|x| format!("{:.4}", x)).collect();
            writeln!(f, "    data: [{}]", vals.join(", "))?;
        }
        write!(f, "}}")
    }
}
