use super::{preprocess, rotated_frames, scaled_frames, window_crop};
use autograph::{
    device::Device,
    result::Result,
    tensor::{float::FloatTensor4, Tensor, Tensor1, TensorView},
};
use image::{open, RgbImage};
use ndarray::{Array1, Array4, ArrayView1, ArrayView4, Axis};
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
use std::fs::{read_dir, File};
use std::io::{self, BufRead};
use std::vec;

pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

pub struct DataSet {
    path: String,
    pub data: Vec<(String, RgbImage)>,
    names: Vec<String>,
    window_size: u32,
}

impl DataSet {
    pub fn new(path: String, label_names_path: String, window_size: u32) -> DataSet {
        DataSet {
            path,
            data: Vec::new(),
            names: Self::load_label_names(label_names_path),
            window_size,
        }
    }

    pub fn load_label_names(path: String) -> Vec<String> {
        let file = File::open(path).unwrap();
        io::BufReader::new(file)
            .lines()
            .map(|line| line.unwrap())
            .collect()
    }

    pub fn load(&mut self, augment: bool) {
        let pathes = Self::list_pathes(&self.path);
        let annotations = Self::load_annotations(pathes, self.window_size, augment);
        self.data = annotations;
    }

    pub fn list_pathes(path: &str) -> Vec<(String, String)> {
        let mut file_pathes = Vec::new();
        for entry in read_dir(path).unwrap() {
            let path = entry.unwrap();
            if path.path().to_str().unwrap().ends_with(".jpg") {
                let image_path = path.path();
                let image_path = image_path.as_path().to_str().unwrap();
                let labels_path = image_path.replace("jpg", "txt");
                file_pathes.push((labels_path.to_string(), image_path.to_string()));
            }
        }
        file_pathes
    }

    pub fn load_annotation(
        image_path: String,
        label: String,
        x: u32,
        y: u32,
        window_size: u32,
    ) -> (String, RgbImage) {
        let img = open(image_path).unwrap().to_rgb8();
        let window = window_crop(&img, window_size, window_size, (x, y));
        (label, window)
    }

    pub fn load_annotations(
        pathes: Vec<(String, String)>,
        window_size: u32,
        augment: bool,
    ) -> Vec<(String, RgbImage)> {
        let mut annotations = Vec::new();
        for path in pathes {
            let file = File::open(path.0).unwrap();
            for line in io::BufReader::new(file).lines() {
                match line {
                    Ok(line) => {
                        let mut l = line.split(" ");
                        let label = l.next().unwrap();
                        let x: u32 = l.next().unwrap().parse().unwrap();
                        let y: u32 = l.next().unwrap().parse().unwrap();
                        match augment {
                            true => {
                                let annotation = Self::load_annotation(
                                    path.1.clone(),
                                    label.to_string(),
                                    x,
                                    y,
                                    window_size,
                                );
                                let frame = annotation.1.clone();
                                let frames = std::iter::once(&frame)
                                    .cloned()
                                    .chain(rotated_frames(&frame))
                                    .chain(scaled_frames(&frame));
                                let augmented_annotations =
                                    frames.map(|f| (annotation.0.clone(), f));
                                annotations.extend(augmented_annotations);
                            }
                            false => {
                                annotations.push(Self::load_annotation(
                                    path.1.clone(),
                                    label.to_string(),
                                    x,
                                    y,
                                    window_size,
                                ));
                            }
                        };
                    }
                    _ => (),
                }
            }
        }
        annotations
    }

    pub fn generate_random_annotations_from_image(
        image: &RgbImage,
        label: String,
        count: usize,
        window_size: u32,
    ) -> Vec<(String, RgbImage)> {
        let mut annotations = Vec::new();
        let mut rng: ThreadRng = rand::thread_rng();

        for _ in 0..count {
            let x = rng.gen_range(0..=image.width());
            let y = rng.gen_range(0..=image.height());
            annotations.push((
                label.to_string(),
                window_crop(&image, window_size, window_size, (x, y)),
            ));
        }
        annotations
    }

    pub fn generate_random_annotations(&mut self, count_each: usize) {
        let pathes = Self::list_pathes(&self.path);
        for (_, image_path) in pathes {
            let img = open(image_path).unwrap().to_rgb8();
            self.data
                .extend(Self::generate_random_annotations_from_image(
                    &img,
                    "none".to_string(),
                    count_each,
                    self.window_size,
                ));
        }
    }

    pub fn label_props(label: &str, labels: &Vec<String>) -> Vec<f32> {
        let mut props = vec![0.0; 10];
        let idx = labels.into_iter().position(|x| x == label).unwrap();
        props[idx] = 1.0;
        props
    }

    pub fn label_id(label: &str, labels: &Vec<String>) -> Vec<u8> {
        let idx = labels.into_iter().position(|x| x == label).unwrap();
        vec![idx as u8]
    }

    pub fn get(&self) -> ((Array4<f32>, Array1<u8>), (Array4<f32>, Array1<u8>)) {
        let (train_x, num_image_train): (Vec<f32>, usize) = (
            self.data
                .iter()
                .flat_map(|(_, img)| preprocess(&img))
                .collect(),
            self.data.len(),
        );
        let (train_y, num_label_train): (Vec<u8>, usize) = (
            self.data
                .iter()
                .flat_map(|(label, _)| Self::label_id(label, &self.names))
                .collect(),
            self.data.len(),
        );
        let (test_x, num_image_test): (Vec<f32>, usize) = (
            self.data
                .iter()
                .flat_map(|(_, img)| preprocess(&img))
                .collect(),
            self.data.len(),
        );
        let (test_y, num_label_test): (Vec<u8>, usize) = (
            self.data
                .iter()
                .flat_map(|(label, _)| Self::label_id(label, &self.names))
                .collect(),
            self.data.len(),
        );

        let x_train = Array4::from_shape_vec(
            (
                num_image_train,
                self.window_size as usize,
                self.window_size as usize,
                3 as usize,
            ),
            train_x,
        )
        .unwrap()
        .into_shape((
            num_image_test,
            3 as usize,
            self.window_size as usize,
            self.window_size as usize,
        ))
        .unwrap();
        let y_train = Array1::from_shape_vec((num_label_train,), train_y).unwrap();
        let x_test = Array4::from_shape_vec(
            (
                num_image_test,
                self.window_size as usize,
                self.window_size as usize,
                3 as usize,
            ),
            test_x,
        )
        .unwrap()
        .into_shape((
            num_image_test,
            3 as usize,
            self.window_size as usize,
            self.window_size as usize,
        ))
        .unwrap();
        let y_test = Array1::from_shape_vec((num_label_test,), test_y).unwrap();
        ((x_train, y_train), (x_test, y_test))
    }

    pub fn samples(&self) -> usize {
        self.data.len()
    }

    pub fn batch_iter<'a>(
        device: &'a Device,
        images: &'a ArrayView4<f32>,
        classes: &'a ArrayView1<u8>,
        batch_size: usize,
    ) -> impl ExactSizeIterator<Item = Result<(FloatTensor4, Tensor1<u8>)>> + 'a {
        images
            .axis_chunks_iter(Axis(0), batch_size)
            .into_iter()
            .zip(classes.axis_chunks_iter(Axis(0), batch_size))
            .map(move |(x, t)| {
                let x = smol::block_on(TensorView::try_from(x)?.into_device(device.clone()))?
                    // normalize the bytes to f32
                    .scale_into::<f32>(1. / 255.)?
                    .into_float();
                let t = smol::block_on(TensorView::try_from(t)?.into_device(device.clone()))?;
                Ok((x, t))
            })
    }

    pub fn shuffled_batch_iter<'a>(
        device: &'a Device,
        images: &'a ArrayView4<'a, f32>,
        classes: &'a ArrayView1<'a, u8>,
        batch_size: usize,
    ) -> impl ExactSizeIterator<Item = Result<(FloatTensor4, Tensor1<u8>)>> + 'a {
        let mut indices = (0..images.shape()[0]).into_iter().collect::<Vec<usize>>();
        indices.shuffle(&mut rand::thread_rng());
        (0..indices.len())
            .into_iter()
            .step_by(batch_size)
            .map(move |index| {
                let batch_indices = &indices[index..(index + batch_size).min(indices.len())];
                let x = batch_indices
                    .iter()
                    .copied()
                    .flat_map(|i| images.index_axis(Axis(0), i))
                    .copied()
                    .collect::<Array1<f32>>()
                    .into_shape([
                        batch_indices.len(),
                        images.dim().1,
                        images.dim().2,
                        images.dim().3,
                    ])?;
                let t = batch_indices
                    .iter()
                    .copied()
                    .map(|i| classes[i])
                    .collect::<Array1<u8>>();
                let x = smol::block_on(Tensor::from(x).into_device(device.clone()))?
                    // normalize the bytes to f32
                    .scale_into::<f32>(1. / 255.)?
                    .into_float();
                let t = smol::block_on(Tensor::from(t).into_device(device.clone()))?;
                Ok((x, t))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LABELS: usize = 18;
    const IMAGES_PER_LABEL: usize = 21;

    #[test]
    fn test_list_files() {
        let path = "res/training";
        let file_pathes = DataSet::list_pathes(path);
        assert_eq!(file_pathes.len(), 3);
    }

    #[test]
    fn test_load_annotations() {
        let pathes = vec![(
            "res/training/webcam01.txt".to_string(),
            "res/training/webcam01.jpg".to_string(),
        )];
        let annotations = DataSet::load_annotations(pathes, 28, false);
        assert_eq!(annotations.len(), 6);
    }

    #[test]
    fn test_dataset() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(false);
        assert_eq!(dataset.samples(), 18);
    }

    #[test]
    fn test_load_label_names() {
        let labels = DataSet::load_label_names("res/labels.txt".to_string());
        assert_eq!(labels.len(), 10);
        assert_eq!(labels[5], "loco5");
        assert_eq!(labels.into_iter().position(|x| x == "loco5"), Some(5));
    }

    #[test]
    fn test_label_props() {
        let labels = DataSet::load_label_names("res/labels.txt".to_string());
        let props = DataSet::label_props("loco5", &labels);
        assert_eq!(props.len(), 10);
        assert_eq!(props[5], 1.0);
        assert_eq!(props[0], 0.0);
    }

    #[test]
    fn test_load_augmented() {
        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL);
    }

    #[test]
    fn test_generate_random_annotations() {
        let image = RgbImage::new(32, 32);
        let annotations =
            DataSet::generate_random_annotations_from_image(&image, "none".to_string(), 5, 28);
        assert_eq!(annotations.len(), 5);
        assert_eq!(annotations.last().unwrap().0, "none");

        let mut dataset = DataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL);
        dataset.generate_random_annotations(1);
        assert_eq!(dataset.samples(), LABELS * IMAGES_PER_LABEL + 3);
    }
}
