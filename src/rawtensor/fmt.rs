use super::RawTensor;

use core::fmt;

fn fmt_matrix(f: &mut fmt::Formatter, values: &[f64], cols: usize, indent: &str) -> fmt::Result {
    let formatted: Vec<Vec<String>> = values.chunks(cols)
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

impl RawTensor {
    // Writes the data section in logical order, indented. Used by Tensor's Display.
    pub(crate) fn fmt_data(&self, f: &mut fmt::Formatter, indent: &str) -> fmt::Result {
        if self.shape.len() == 2 {
            let logical: Vec<f64> = self.iter().collect();
            fmt_matrix(f, &logical, self.shape[1], indent)
        } else {
            let vals: Vec<String> = self.iter().map(|x| format!("{:.4}", x)).collect();
            writeln!(f, "{}[{}]", indent, vals.join(", "))
        }
    }
}

// Brute-force: shape, strides, and raw backing data in memory order
impl fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vals: Vec<String> = self.data.iter().map(|x| format!("{:.4}", x)).collect();
        write!(f, "RawTensor {{ shape: {:?}, strides: {:?}, data: [{}] }}",
            self.shape, self.strides, vals.join(", "))
    }
}

// Logical order: shape and data. 2D tensors are shown as a matrix.
impl fmt::Display for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "RawTensor {{")?;
        writeln!(f, "    shape: {:?}", self.shape)?;
        if self.shape.len() == 2 { writeln!(f, "    data:")?; }
        self.fmt_data(f, "        ")?;
        write!(f, "}}")
    }
}
