use tensorpack::Tensor;

fn main() {
    let x: Tensor = Tensor::linspace(1.0, 12.0, 12).requires_grad();
    let x = x.reshape(&[3, 4]);
    x.backward();
    println!("1-dim debug {:?}", x);
    println!("1-dim pretty {}", x);
    println!("\n\n\n");
    // println!("2-dim debug {:?}", x);
    // println!("2-dim pretty {}", x);
    // println!("\n\n\n");
    // x.reshape(&[2, 2, 3]);
    // println!("3-dim debug {:?}", x);
    // println!("3-dim pretty {}", x);
    // println!("\n\n\n");
    //
    let x: Tensor = Tensor::linspace(1.0, 6.0, 12);
    let y: Tensor = Tensor::linspace(1.0, 6.0, 12);
    println!("{:?}\n{:?}\n{:?}", x, y, &x + &y);

    Tensor::set_seed(21);
    let a: Tensor = Tensor::rand_range(&[6], 0.0, 5.0);

    Tensor::set_seed(52);
    let b: Tensor = Tensor::rand_range(&[6], 0.0, 5.0);

    Tensor::set_seed(21);
    let c: Tensor = Tensor::rand_range(&[6], 0.0, 5.0);
    println!("{:?}\n{:?}\n{:?}\n", a, b, c);
}
