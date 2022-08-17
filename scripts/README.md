# Train the model

## Create virtualenv

```sh
virtualenv -p python3 .venv
source .venv/bin/activate
pip install pandas tensorflow pillow numpy tf2onnx
```

## Train

```sh
python train.py
```

## Export Model

```sh
python export_onnx_model.py
cp mobilenetv2.onnx ../res/
```
