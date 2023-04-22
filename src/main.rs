use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Tensor};


fn test1() {
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
}

fn net(vs: &nn::Path,dim_in:i64,dim_hid:i64,dim_out:i64) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            dim_in,
            dim_hid,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, dim_hid, dim_out, Default::default()))
}

fn my_module(p: &nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.randn_standard("x1", &[dim]);
    let x2 = p.randn_standard("x2", &[dim]);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

fn gradient_descent() {
    let vs = nn::VarStore::new(Device::Cpu);
//    let my_module = my_module(&vs.root(), 7);
    let my_module = net(&vs.root(), 7, 14, 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
//    let xs = Tensor::zeros(&[7], kind::FLOAT_CPU);
    let xs = Tensor::of_slice(&[0.1_f32,0.2,0.3,0.4,0.5,0.6,0.7]);
    println!("xs:");
    xs.print();
    let ys = Tensor::zeros(&[7], kind::FLOAT_CPU);
    println!("ys:");
    ys.print();
    for idx in 1..1000 {
        // Dummy mini-batches made of zeros.
	let res = my_module.forward(&xs);
        let loss = (&res - &ys).pow_tensor_scalar(2).sum(kind::Kind::Float);
	if idx%100 == 99 {
	    println!("Res:");
	    res.print();
	    println!("Loss:");
	    loss.print();
	}
        opt.backward_step(&loss);
    }
}

fn gradient_descent2() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = net(&vs.root(), 7, 14, 1);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    let xs = Tensor::of_slice2(&[
	[0.1_f32,0.2,0.3,0.4,0.5,0.6,0.7],
	[0.2_f32,0.4,0.6,0.8,1.0,1.2,1.4]
    ]);
    println!("xs:");
    xs.print();
    let ys = Tensor::of_slice2(&[
	[0.1_f32],
	[0.2_f32]
    ]);
    println!("ys:");
    ys.print();
    for idx in 1..10000 {
        // Dummy mini-batches made of zeros.
	let res = my_module.forward(&xs);
        let loss = (&res - &ys).pow_tensor_scalar(2).sum(kind::Kind::Float);
	if idx%1000 == 999 {
	    println!("Res:");
	    res.print();
	    println!("Loss:");
	    loss.print();
	}
        opt.backward_step(&loss);
    }
}

fn gradient_descent3() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = net(&vs.root(), 2, 14, 1);
    let mut opt = nn::Sgd::default().build(&vs, 1e-3).unwrap();
    let xs = Tensor::of_slice2(&[
	[0.1_f32,0.1],
	[0.1_f32,0.2]
    ]);
    println!("xs:");
    xs.print();
    let ys = Tensor::of_slice2(&[
	[0.01_f32],
	[0.04_f32]
    ]);
    println!("ys:");
    ys.print();
    for idx in 1..1000000 {
        // Dummy mini-batches made of zeros.
	let res = my_module.forward(&xs);
        let loss = (&res - &ys).pow_tensor_scalar(2).sum(kind::Kind::Float);
	if idx%10000 == 9999 {
	    println!("ys:");
	    ys.print();
	    println!("Res:");
	    res.print();
	    println!("Loss:");
	    loss.print();
	}
        opt.backward_step(&loss);
    }
}

fn main() {
    gradient_descent3();
}
