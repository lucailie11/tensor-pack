use rml::Tensor;

fn main() {
    let x = Tensor::linspace(1.0, 24.0, 24);
    let y = Tensor::linspace(1.0, 24.0, 24);

    let z: Tensor = &x + &y;

    println!("{}\n{}\n{}\n", x, y, z);
}
