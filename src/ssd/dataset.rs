use ag::ndarray;
use autograd as ag;
use image::{open, GrayImage};
use mosse::utils::{preprocess, window_crop};
use std::fs::read_dir;
use std::fs::File;
use std::io::{self, BufRead};
use std::vec;

pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

pub struct DataSet {
    path: String,
    data: Vec<(String, GrayImage)>,
    names: Vec<String>,
}

impl DataSet {
    pub fn new(path: String, label_names_path: String) -> DataSet {
        DataSet {
            path,
            data: Vec::new(),
            names: Self::load_label_names(label_names_path),
        }
    }

    pub fn load_label_names(path: String) -> Vec<String> {
        let file = File::open(path).unwrap();
        io::BufReader::new(file)
            .lines()
            .map(|line| line.unwrap())
            .collect()
    }

    pub fn load(&mut self) {
        let pathes = Self::list_pathes(&self.path);
        let annotations = Self::load_annotations(pathes);
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
    ) -> (String, GrayImage) {
        let img = open(image_path).unwrap().to_luma8();
        let window = window_crop(&img, 28, 28, (x, y));
        (label, window)
    }

    pub fn load_annotations(pathes: Vec<(String, String)>) -> Vec<(String, GrayImage)> {
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
                        annotations.push(Self::load_annotation(
                            path.1.clone(),
                            label.to_string(),
                            x,
                            y,
                        ))
                    }
                    _ => (),
                }
            }
        }
        annotations
    }

    pub fn label_props(label: &str, labels: &Vec<String>) -> Vec<f32> {
        let mut props = vec![0.0; 10];
        let idx = labels.into_iter().position(|x| x == label).unwrap();
        props[idx] = 1.0;
        props
    }

    pub fn get(&self) -> ((NdArray, NdArray), (NdArray, NdArray)) {
        let (train_x, num_image_train): (Vec<f32>, usize) = (
            self.data
                .iter()
                .flat_map(|(_, img)| preprocess(&img))
                .collect(),
            self.data.len(),
        );
        let (train_y, num_label_train): (Vec<f32>, usize) = (
            self.data
                .iter()
                .flat_map(|(label, _)| Self::label_props(label, &self.names))
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
        let (test_y, num_label_test): (Vec<f32>, usize) = (
            self.data
                .iter()
                .flat_map(|(label, _)| Self::label_props(label, &self.names))
                .collect(),
            self.data.len(),
        );

        // Vec to ndarray
        let as_arr = NdArray::from_shape_vec;
        let x_train = as_arr(ndarray::IxDyn(&[num_image_train, 28 * 28]), train_x).unwrap();
        let y_train = as_arr(ndarray::IxDyn(&[num_label_train, 1]), train_y).unwrap();
        let x_test = as_arr(ndarray::IxDyn(&[num_image_test, 28 * 28]), test_x).unwrap();
        let y_test = as_arr(ndarray::IxDyn(&[num_label_test, 1]), test_y).unwrap();
        ((x_train, y_train), (x_test, y_test))
    }

    pub fn samples(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_files() {
        let path = "res/training";
        let file_pathes = DataSet::list_pathes(path);
        assert_eq!(file_pathes.len(), 2);
    }

    #[test]
    fn test_load_annotations() {
        let pathes = vec![(
            "res/training/webcam01.txt".to_string(),
            "res/training/webcam01.jpg".to_string(),
        )];
        let annotations = DataSet::load_annotations(pathes);
        assert_eq!(annotations.len(), 4);
    }

    #[test]
    fn test_dataset() {
        let mut dataset = DataSet::new("res/training/".to_string(), "res/labels.txt".to_string());
        dataset.load();
        assert_eq!(dataset.samples(), 8);
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
}
