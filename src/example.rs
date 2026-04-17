use rml::Tensor;

fn main() {
    let mut x: Tensor = Tensor::linspace(1.0, 6.0, 12);
    println!("1-dim debug {:?}", x);
    println!("1-dim pretty {}", x);
    println!("\n\n\n");
    x.reshape(&[3, 4]);
    println!("2-dim debug {:?}", x);
    println!("2-dim pretty {}", x);
    println!("\n\n\n");
    x.reshape(&[2, 2, 3]);
    println!("3-dim debug {:?}", x);
    println!("3-dim pretty {}", x);
    println!("\n\n\n");
}

