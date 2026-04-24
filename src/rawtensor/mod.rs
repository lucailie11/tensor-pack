mod accumulate;     // handles accumulations function for logic in grad/
mod core;           // declares the RawTensor type and implements basic functions
mod iter;           // creates iterators in logical order
mod binary;         // handles elementwise arihemtics between two RawTensors
mod constructors;   // implements various constructors (e.g. from_slice, randn, linspace)
mod linalg;         // handles matmul and dotproduct
mod fmt;            // implements Debug and Display traits for RawTensor
mod normalizations; // handles normalization operations on an axis (e.g. softmax)
mod reductions;     // handles reduction operations on an axis (sum, mean, var, std_dev, etc)
mod scalar;         // handles arithmetics between a RawTensor and a scalar(f64)
mod structure;      // handles stuff related to the shape of a RawTensor (indexing, broadcasting,
                    // expanding, squeezing, reshaping, tranposing etc)
mod unary;          // handles maps and unary operations on RawTensor (exp, ln, etc)

pub(crate) use core::RawTensor;
pub(crate) use fmt::fmt_matrix;
