use crate::bbox::BBox;
use crate::dataset::{rotated_frames, scaled_frames, window_crop, DataSet};
use crate::Detector;
use image::{open, RgbImage};
use rand::prelude::ThreadRng;
use rand::Rng;
use std::fs::read_dir;
use std::fs::File;
use std::io::{self, BufRead};
use std::vec;

pub struct FolderDataSet {
    path: String,
    pub data: Vec<(String, RgbImage)>,
    names: Vec<String>,
    window_size: u32,
}

impl FolderDataSet {
    pub fn new(path: String, label_names_path: String, window_size: u32) -> Self {
        FolderDataSet {
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
                        let mut l = line.split(' ');
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

    pub fn get_negative_samples(
        &self,
        detector: &dyn Detector,
        class: u32,
    ) -> Vec<(String, RgbImage)> {
        let mut annotations = Vec::new();
        let pathes = Self::list_pathes(&self.path);
        let class_label = self.names[class as usize].clone();
        for path in pathes {
            let file = File::open(path.0).unwrap();
            let mut pos_bboxes = Vec::new();
            for line in io::BufReader::new(file).lines() {
                match line {
                    Ok(line) => {
                        let mut l = line.split(' ');
                        let label = l.next().unwrap();
                        if label == class_label {
                            let x: u32 = l.next().unwrap().parse().unwrap();
                            let y: u32 = l.next().unwrap().parse().unwrap();

                            let bbox = BBox {
                                x: x as f32,
                                y: y as f32,
                                w: self.window_size as f32,
                                h: self.window_size as f32,
                            };
                            pos_bboxes.push(bbox);
                        }
                    }
                    _ => (),
                }
            }

            let img_path = path.1;
            let img = open(img_path.clone()).unwrap();
            let detections = detector.detect_objects(&img);

            detections.iter().for_each(|detection| {
                let mut false_pos = true;
                pos_bboxes.iter().for_each(|bbox| {
                    if bbox.iou(&detection.bbox) > 0.1 {
                        false_pos = false;
                    }
                });
                if false_pos {
                    annotations.push(Self::load_annotation(
                        img_path.clone(),
                        "none".to_string(),
                        detection.bbox.x as u32,
                        detection.bbox.y as u32,
                        self.window_size,
                    ));
                }
            });
        }
        annotations
    }

    pub fn generate_hard_negative_samples(&mut self, detector: &dyn Detector, class: u32) {
        let annotations = self.get_negative_samples(detector, class);
        self.data.extend(annotations);
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
                window_crop(image, window_size, window_size, (x, y)),
            ));
        }
        annotations
    }

    pub fn label_props(label: &str, labels: &[String]) -> Vec<f32> {
        let mut props = vec![0.0; 10];
        let idx = labels.iter().position(|x| x == label).unwrap();
        props[idx] = 1.0;
        props
    }

    pub fn label_id(label: &str, labels: &[String]) -> u32 {
        labels.iter().position(|x| x == label).unwrap() as u32
    }

    pub fn export(&self, folder: &str) {
        self.data
            .iter()
            .enumerate()
            .for_each(|(index, (label, img))| {
                img.save(format!("{}/{}_{}.jpg", folder, label, index))
                    .unwrap();
            });
    }
}

impl DataSet for FolderDataSet {
    fn load(&mut self, augment: bool) {
        let pathes = Self::list_pathes(&self.path);
        let annotations = Self::load_annotations(pathes, self.window_size, augment);
        self.data = annotations;
    }

    fn generate_random_annotations(&mut self, count_each: usize) {
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

    fn get(&self) -> (Vec<RgbImage>, Vec<u32>, Vec<RgbImage>, Vec<u32>) {
        let train_x = self.data.iter().map(|(_, img)| img.clone()).collect();
        let train_y = self
            .data
            .iter()
            .map(|(label, _)| Self::label_id(label, &self.names))
            .collect();

        let test_x = self.data.iter().map(|(_, img)| img.clone()).collect();
        let test_y = self
            .data
            .iter()
            .map(|(label, _)| Self::label_id(label, &self.names))
            .collect();
        (train_x, train_y, test_x, test_y)
    }

    fn samples(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANNOTATIONS: usize = 18;
    const IMAGES_PER_LABEL: usize = 21;

    #[test]
    fn test_list_files() {
        let path = "res/training";
        let file_pathes = FolderDataSet::list_pathes(path);
        assert_eq!(file_pathes.len(), 3);
    }

    #[test]
    fn test_load_annotations() {
        let pathes = vec![(
            "res/training/webcam01.txt".to_string(),
            "res/training/webcam01.jpg".to_string(),
        )];
        let annotations = FolderDataSet::load_annotations(pathes, 28, false);
        assert_eq!(annotations.len(), 6);
    }

    #[test]
    fn test_dataset() {
        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(false);
        assert_eq!(dataset.samples(), ANNOTATIONS);
    }

    #[test]
    fn test_load_label_names() {
        let labels = FolderDataSet::load_label_names("res/labels.txt".to_string());
        assert_eq!(labels.len(), 10);
        assert_eq!(labels[5], "loco5");
        assert_eq!(labels.into_iter().position(|x| x == "loco5"), Some(5));
    }

    #[test]
    fn test_label_props() {
        let labels = FolderDataSet::load_label_names("res/labels.txt".to_string());
        let props = FolderDataSet::label_props("loco5", &labels);
        assert_eq!(props.len(), 10);
        assert_eq!(props[5], 1.0);
        assert_eq!(props[0], 0.0);
    }

    #[test]
    fn test_load_augmented() {
        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), ANNOTATIONS * IMAGES_PER_LABEL);
    }

    #[test]
    fn test_generate_random_annotations() {
        let image = RgbImage::new(32, 32);
        let annotations = FolderDataSet::generate_random_annotations_from_image(
            &image,
            "none".to_string(),
            5,
            28,
        );
        assert_eq!(annotations.len(), 5);
        assert_eq!(annotations.last().unwrap().0, "none");

        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        assert_eq!(dataset.samples(), ANNOTATIONS * IMAGES_PER_LABEL);
        dataset.generate_random_annotations(1);
        assert_eq!(dataset.samples(), ANNOTATIONS * IMAGES_PER_LABEL + 3);
    }

    #[test]
    fn test_dataset_export() {
        let mut dataset = FolderDataSet::new(
            "res/training/".to_string(),
            "res/labels.txt".to_string(),
            28,
        );
        dataset.load(true);
        dataset.export("out/export");
    }
}
