use super::preprocess;
use super::NdArray;
use crate::dataset::{DataSet as _, FolderDataSet};
use ag::ndarray;
use autograd as ag;

pub struct DataSet {
    pub dataset: FolderDataSet,
}

impl DataSet {
    pub fn new(path: String, label_names_path: String, window_size: u32) -> DataSet {
        let dataset = FolderDataSet::new(path, label_names_path, window_size);
        DataSet { dataset }
    }

    pub fn load(&mut self, augment: bool) {
        self.dataset.load(augment);
    }

    pub fn generate_random_annotations(&mut self, count_each: usize) {
        self.dataset.generate_random_annotations(count_each)
    }

    pub fn get(&self) -> ((NdArray, NdArray), (NdArray, NdArray)) {
        let (train_x, train_y, test_x, test_y) = self.dataset.get();
        train_x.iter().for_each(|img| assert_eq!(img.width(), img.height()));
        let window_size = self.dataset.window_size;
        train_x.iter().for_each(|img| assert_eq!(img.width(), window_size));

        let (train_x, num_image_train): (Vec<f32>, usize) = (
            train_x.iter().flat_map(|img| preprocess(&img)).collect(),
            train_x.len(),
        );
        let train_y = train_y.iter().map(|y| *y as f32).collect();
        let (test_x, num_image_test): (Vec<f32>, usize) = (
            test_x.iter().flat_map(|img| preprocess(&img)).collect(),
            test_x.len(),
        );
        let test_y = test_y.iter().map(|y| *y as f32).collect();

        // Vec to ndarray
        let as_arr = NdArray::from_shape_vec;
        println!("{:?}", num_image_train);
        let x_train = as_arr(
            ndarray::IxDyn(&[
                num_image_train,
                (self.dataset.window_size * self.dataset.window_size * 3) as usize,
            ]),
            train_x,
        )
        .unwrap();
        let y_train = as_arr(ndarray::IxDyn(&[num_image_train, 1]), train_y).unwrap();
        let x_test = as_arr(
            ndarray::IxDyn(&[
                num_image_test,
                (self.dataset.window_size * self.dataset.window_size * 3) as usize,
            ]),
            test_x,
        )
        .unwrap();
        let y_test = as_arr(ndarray::IxDyn(&[num_image_test, 1]), test_y).unwrap();
        ((x_train, y_train), (x_test, y_test))
    }

    pub fn samples(&self) -> usize {
        self.dataset.samples()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANNOTATIONS: usize = 97;

    #[test]
    fn test_get() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(false);
        assert_eq!(dataset.samples(), ANNOTATIONS);
        let ((train_x, train_y), (test_x, test_y)) = dataset.get();
        assert_eq!(train_x.shape()[0], ANNOTATIONS);
        assert_eq!(train_y.shape()[0], ANNOTATIONS);
        assert_eq!(test_x.shape()[0], ANNOTATIONS);
        assert_eq!(test_y.shape()[0], ANNOTATIONS);
    }
}
