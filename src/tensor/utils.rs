pub fn stride_len(data: &[f64], step: usize) -> usize {
    (data.len() - 1) / step + 1
}
