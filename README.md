# cam-pos-system

cam-pos-system is a Rust library with an example app that reads a video stream and detects moving objects within a coordinate system using computer vision algorithms.

![moving train](res/red_train.gif)

![Activity](https://www.plantuml.com/plantuml/proxy?cache=no&src=https://raw.github.com/rusty-rails/cam-pos-system/main/docs/activity.puml)

## Algorithms

The cam-pos-system library uses the following computer vision algorithms for object detection:

* HOG detector: This algorithm detects the presence of objects in an image by looking for their Histogram of Oriented Gradients (HOG) features.
* MosseTracker: This algorithm tracks a moving object in a video stream using adaptive correlation filters to estimate the object's position and scale.
The library combines these algorithms to detect moving objects in a video stream and estimate their positions within a coordinate system.

## run wasm

```sh
wasm-pack build --no-default-features --target web
python3 -m http.server
```

## training

In the folder res/training are images. For each image exists a txt file containing labels in format "name x y" The label names can be found in res/labels.txt.

### annotator tool

You can create the training data with this simple tool on [Hog-Detector](https://chriamue.github.io/hog-detector).
