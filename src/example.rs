use rml::Tensor;

fn main() {
    let x = Tensor::linspace(1.0, 24.0, 24);
    let y = Tensor::linspace(1.0, 24.0, 24);

    println!("{}\n{}\n", x, x.matmul(y));
}
