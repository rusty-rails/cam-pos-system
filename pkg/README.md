# cam-pos-system

## run wasm

```sh
wasm-pack build --no-default-features --target web
python3 -m http.server
```

## training

In the folder res/training are images. For each image exists a txt file containing labels in format "name x y" The label names can be found in res/labels.txt.