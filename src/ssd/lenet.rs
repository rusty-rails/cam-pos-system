// source: https://github.com/charles-r-earp/autograph/blob/main/examples/neural-network-mnist/src/main.rs

use autograph::{
    learn::neural_network::layer::{Conv, Dense, Forward, Layer, MaxPool, Relu},
    learn::neural_network::Network,
    result::Result,
};
use serde::{Deserialize, Serialize};

#[derive(Layer, Forward, Clone, Debug, Serialize, Deserialize)]
pub struct Lenet5 {
    #[autograph(layer)]
    conv1: Conv,
    #[autograph(layer)]
    relu1: Relu,
    #[autograph(layer)]
    pool1: MaxPool,
    #[autograph(layer)]
    conv2: Conv,
    #[autograph(layer)]
    relu2: Relu,
    #[autograph(layer)]
    pool2: MaxPool,
    #[autograph(layer)]
    dense1: Dense,
    #[autograph(layer)]
    relu3: Relu,
    #[autograph(layer)]
    dense2: Dense,
    #[autograph(layer)]
    relu4: Relu,
    #[autograph(layer)]
    dense3: Dense,
}

impl Lenet5 {
    fn new() -> Result<Self> {
        let conv1 = Conv::from_inputs_outputs_kernel(3, 6, [5, 5]);
        let relu1 = Relu::default();
        let pool1 = MaxPool::from_kernel([2, 2]).with_strides(2)?;
        let conv2 = Conv::from_inputs_outputs_kernel(6, 16, [5, 5]);
        let relu2 = Relu::default();
        let pool2 = MaxPool::from_kernel([2, 2]).with_strides(2)?;
        let dense1 = Dense::from_inputs_outputs(16 * 4 * 4, 120);
        let relu3 = Relu::default();
        let dense2 = Dense::from_inputs_outputs(120, 84);
        let relu4 = Relu::default();
        let dense3 = Dense::from_inputs_outputs(84, 10).with_bias(true)?;
        Ok(Self {
            conv1,
            relu1,
            pool1,
            conv2,
            relu2,
            pool2,
            dense1,
            relu3,
            dense2,
            relu4,
            dense3,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Lenet {
    pub net: Network<Lenet5>,
    pub input_width: usize,
    pub input_height: usize,
}

impl Lenet {
    pub fn new(input_width: usize, input_height: usize) -> Lenet {
        let net = Network::from(Lenet5::new().unwrap());

        Lenet {
            net,
            input_width,
            input_height,
        }
    }
}

impl Default for Lenet {
    fn default() -> Self {
        Lenet::new(28, 28)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let lenet = Lenet::default();
        println!("{:?}", lenet);
    }
}
