// General-purpose helper functions that support tensor operations without containing any tensor logic themselves.

// Returns the number of elements reachable from `data` when stepping by `step`.
// Equivalent to ceil(data.len() / step).
pub fn stride_len(data: &[f64], step: usize) -> usize {
    (data.len() - 1) / step + 1
}
