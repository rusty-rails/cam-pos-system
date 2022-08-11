// source: https://github.com/raskr/rust-autograd/blob/master/examples/cnn_mnist.rs
use autograph::{
    learn::neural_network::layer::{Conv, Dense, Forward, Layer, Relu},
    result::Result,
};
use serde::{Deserialize, Serialize};

#[derive(Layer, Forward, Clone, Debug, Serialize, Deserialize)]
pub struct CNN {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    dense2: Dense,
}

impl CNN {
    fn new() -> Result<Self> {
        let conv1 = Conv::from_inputs_outputs_kernel(3, 6, [5, 5]);
        let relu1 = Relu::default();
        let dense1 = Dense::from_inputs_outputs(6 * 24 * 24, 84);
        let relu2 = Relu::default();
        let dense2 = Dense::from_inputs_outputs(84, 10).with_bias(true)?;
        Ok(Self {
            conv1,
            relu1,
            dense1,
            relu2,
            dense2,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MobileNet {
    pub net: CNN,
}

impl MobileNet {
    pub fn new() -> MobileNet {
        MobileNet {
            net: CNN::new().unwrap(),
        }
    }
}

impl Default for MobileNet {
    fn default() -> Self {
        MobileNet::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let mobilenet = MobileNet::default();
        println!("{:?}", mobilenet);
    }
}
