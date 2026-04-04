// Broadcasting utilities for shape inference and index mapping.

// Checks whether two dimensions are compatible under broadcasting rules.
fn are_dimensions_compatible(d1: usize, d2: usize) -> bool {
    (d1 == d2) || (d1 == 1) || (d2 == 1)
}

// Computes the output shape from two broadcastable shapes. Panics if they aren't compatible.
pub fn get_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let len = usize::max(shape1.len(), shape2.len());
    let mut out = vec![1usize; len];
    for i in 0..len {
        let d1 = if i < shape1.len() { shape1[shape1.len() - 1 - i] } else { 1 };
        let d2 = if i < shape2.len() { shape2[shape2.len() - 1 - i] } else { 1 };
        assert!(are_dimensions_compatible(d1, d2), "Shapes are not broadcastable");
        out[len - 1 - i] = usize::max(d1, d2);
    }
    out
}

// Maps a flat output index to the corresponding flat index in the input shape, accounting for broadcasting.
pub fn get_broadcast_index(out_index: usize, in_shape: &[usize], out_shape: &[usize]) -> usize {
    let mut in_index: usize = 0;
    let mut in_inner_size: usize = 1;
    let mut out_inner_size: usize = 1;
    for (i, x) in out_shape.iter().rev().enumerate() {
        if i < in_shape.len() && in_shape[in_shape.len() - 1 - i] != 1 {
            in_index += out_index % (out_inner_size * x) / out_inner_size * in_inner_size;
            in_inner_size *= x;
        }
        out_inner_size *= x;
    }
    in_index
}
