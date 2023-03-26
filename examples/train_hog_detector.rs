#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("dataset example does not run in wasm32 mode");
}
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let (width, height) = (32, 32);
    use hog_detector::classifier::BayesClassifier;
    use hog_detector::hogdetector::HogDetectorTrait;
    use hog_detector::{DataSet, HogDetector};
    use image::imageops::FilterType;
    use object_detector_rust::dataset::FolderDataSet;
    use object_detector_rust::detector::PersistentDetector;
    use std::fs::File;

    let mut model: HogDetector<f32, usize, BayesClassifier<f32, usize>, _> = HogDetector::default();

    let data_path = std::fs::canonicalize("res/training/").unwrap();
    let labels_path = std::fs::canonicalize("res/labels.txt").unwrap();
    let label_names = FolderDataSet::load_label_names(labels_path.to_str().unwrap());
    let mut dataset = FolderDataSet::new(data_path.to_str().unwrap(), width, height, label_names);

    dataset.load().unwrap();
    
    let (x, y) = dataset.get_data();
    let x = x
        .into_iter()
        .map(|x| x.resize_exact(width, height, FilterType::Gaussian))
        .collect::<Vec<_>>();
    let y = y.into_iter().map(|y| y as usize).collect::<Vec<_>>();
    model.fit_class(&x, &y, 5).unwrap();
    println!("evaluated: {:?} %", model.evaluate(&dataset, 5) * 100.0);

    let file_writer = File::create("out/hog_bayes_model.json").unwrap();

    model.save(file_writer).unwrap();
}
