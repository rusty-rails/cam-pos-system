// source: https://github.com/jjhbw/mosse-tracker
use image::GrayImage;
use mosse::{self, MosseTracker, MosseTrackerSettings, Prediction};
use std::fs;

pub struct ObjectDetection {
    tracker: MosseTracker,
}

impl ObjectDetection {
    pub fn new(path: String) -> ObjectDetection {
        let (width, height) = (512, 512);
        let window_size = 16;
        let psr_threshold = 7.0;
        let settings = MosseTrackerSettings {
            window_size,
            width,
            height,
            regularization: 0.002,
            learning_rate: 0.05,
            psr_threshold,
        };
        let mut tracker = MosseTracker::new(&settings);

        let mut train_images = Vec::new();

        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap();
            if path.path().to_str().unwrap().ends_with(".jpg") {
                let img = image::open(path.path()).unwrap().to_luma8();
                train_images.push(img);
            }
        }
        let t = train_images.iter().map(|f| f).collect();
        tracker.train(t);

        ObjectDetection { tracker }
    }

    pub fn predict(&self, frame: &GrayImage) -> Prediction {
        self.tracker.predict(frame)
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use approx::assert_relative_eq;
    use image::open;

    #[test]
    fn tracker() {
        let loco01 = open("res/loco01.jpg").unwrap().to_luma8();
        let loco02 = open("res/loco02.jpg").unwrap().to_luma8();
        let loco03 = open("res/loco03.jpg").unwrap().to_luma8();
        let loco04 = open("res/loco04.jpg").unwrap().to_luma8();
        let loco05 = open("res/loco05.jpg").unwrap().to_luma8();

        let webcam1 = open("res/webcam01.jpg").unwrap().to_luma8();
        let webcam6 = open("res/webcam05.jpg").unwrap().to_luma8();

        let (width, height) = webcam1.dimensions();
        let window_size = 16;
        let psr_threshold = 7.0;
        let settings = MosseTrackerSettings {
            window_size,
            width,
            height,
            regularization: 0.002,
            learning_rate: 0.05,
            psr_threshold,
        };

        let mut tracker = MosseTracker::new(&settings);
        tracker.train(vec![&loco01, &loco02, &loco03, &loco04, &loco05]);
        let prediction = tracker.predict(&webcam1);
        println!("{:?}", prediction);

        assert_relative_eq!(prediction.location.0 as f32, 250.0, epsilon = 200.0);
        assert_relative_eq!(prediction.location.1 as f32, 400.0, epsilon = 200.0);
        let prediction = tracker.predict(&webcam6);
        println!("{:?}", prediction);
        assert_relative_eq!(prediction.location.0 as f32, 100.0, epsilon = 400.0);
        assert_relative_eq!(prediction.location.1 as f32, 400.0, epsilon = 400.0);
    }

    #[test]
    fn object_detection() {
        let detector = ObjectDetection::new("res/loco5/".to_string());

        let webcam1 = open("res/webcam01.jpg").unwrap().to_luma8();
        let webcam6 = open("res/webcam05.jpg").unwrap().to_luma8();

        let prediction = detector.predict(&webcam1);
        println!("{:?}", prediction);

        assert_relative_eq!(prediction.location.0 as f32, 250.0, epsilon = 200.0);
        assert_relative_eq!(prediction.location.1 as f32, 400.0, epsilon = 200.0);
        let prediction = detector.predict(&webcam6);
        println!("{:?}", prediction);
        assert_relative_eq!(prediction.location.0 as f32, 100.0, epsilon = 400.0);
        assert_relative_eq!(prediction.location.1 as f32, 400.0, epsilon = 400.0);
    }
}
