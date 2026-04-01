mod tensor;
use tensor::Tensor;

fn main() {
    let mut x: Tensor = Tensor::linspace(0.0, 10.0, 30);
    x.reshape(&[6, 5]);

    println!("{}\n{:?}\n", x, x.sum_axis(0));
}
