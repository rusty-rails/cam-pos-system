import logging
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import random

from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras import Model

CLASS_NAMES = ['none','marker1','marker2','marker3','loco4','loco5','object6','object7','object8','object9']

img_height = 96 # 224
img_width = 96 # 224
batch_size = 30
epochs = 10

data_dir = '../res/training'

class DataSetCreator(object):
    def __init__(self, batch_size, image_height, image_width, data_dir):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.data_dir = data_dir
        x = [x for x in os.listdir(data_dir) if x.endswith(".jpg")]
        y = [y for y in os.listdir(data_dir) if y.endswith(".txt")]
        self.dataset = zip(x,y)
        
    def _get_class(self, label):
        c = list(map( lambda x: 1 if x == label else 0, CLASS_NAMES))
        return c
    
    def _load_image(self, pil_img):
        image = tf.keras.preprocessing.image.img_to_array(pil_img)
        #image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.image.resize(image, [self.image_height, self.image_width])
    
    def _load_labeled_data(self, files):
        data = []
        with open(os.path.join(data_dir, files[1])) as f:
            lines = [line.rstrip() for line in f]
            for line in lines:
                line = line.split(' ')
                label = line[0]
                x = int(line[1])
                y = int(line[2])
                im = PIL.Image.open(os.path.join(data_dir, files[0]))
                left = x - img_width / 2
                top = y - img_height / 2
                right = x + img_width / 2
                bottom = y + img_height / 2
                im = im.crop((left, top, right, bottom))
                im = np.array(im)
                im = im.reshape(1,img_width,img_height,3)
                im = im.astype(np.float32)
                label = self._get_class(label)
                label = np.array(label)
                label = label.reshape(1,10)
                data.append((im, label))
        return data

    def __len__(self):
        return len(self.loaded_dataset)

    def __getitem__(self,idx):
        (x, y) = self.loaded_dataset[idx]
        return x, y
    
    def load_process(self, shuffle_size = 1000):
        self.loaded_dataset = []
        for dataset in self.dataset:
            self.loaded_dataset.extend(self._load_labeled_data(dataset))
        #self.loaded_dataset = self.dataset.map(self._load_labeled_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        #self.loaded_dataset = self.loaded_dataset.cache()

        # Shuffle data and create batches
        #self.loaded_dataset = self.loaded_dataset.shuffle(buffer_size=shuffle_size)
        #self.loaded_dataset = self.loaded_dataset.repeat()
        #self.loaded_dataset = self.loaded_dataset.batch(self.batch_size)

        # Make dataset fetch batches in the background during the training of the model.
        #self.loaded_dataset = self.loaded_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        
    def get_batch(self):
        return next(iter(self.loaded_dataset))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.get_batch()
        except IndexError:
            raise StopIteration
        return result

    def __call__(self):
        for i in range(self.__len__()):
            (x, y) = self.__getitem__(i)
            yield x, y
            
            if i == self.__len__()-1:
                self.on_epoch_end()

    def on_epoch_end(self):
        reidx = random.sample(population = list(range(self.__len__())),k = self.__len__())
        #self.loaded_dataset = self.loaded_dataset[reidx]

o_t = (tf.float32, tf.int64)
o_s = (tf.TensorShape([None, img_width,img_height,3]),tf.TensorShape([None, len(CLASS_NAMES)]))

dg = DataSetCreator(32, 32, 32, data_dir)
dg.load_process()
train_ds = tf.data.Dataset.from_generator(dg, output_types=o_t, output_shapes=o_s)

dg = DataSetCreator(32, 32, 32, data_dir)
dg.load_process()
valid_ds = tf.data.Dataset.from_generator(dg, output_types=o_t, output_shapes=o_s)

dg = DataSetCreator(32, 32, 32, data_dir)
dg.load_process()
test_ds = tf.data.Dataset.from_generator(dg, output_types=o_t, output_shapes=o_s)


def normalize(img, label):
    return img / 255.0, label


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2)
])

train_dataset = (train_ds
                 .map(normalize)
                 .map(lambda x, y: (data_augmentation(x), y))
                 .prefetch(tf.data.AUTOTUNE))

valid_dataset = valid_ds.map(normalize)
test_dataset = test_ds.map(normalize)

print(train_dataset)


def get_mobilenet():
    pre_trained_model = MobileNetV2(
        include_top=False,
        input_shape=(img_height, img_width, 3),
        classifier_activation='softmax'
    )

    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.output
    last_layer.trainable = True

    x = GlobalAveragePooling2D()(last_layer)
    x = Dense(1024, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)
    return model


model = get_mobilenet()
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
checkpoint_path = "./checkpoints/mobilenet/"

#model.load_weights(checkpoint_path)

model_history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=[
        #tf.keras.callbacks.EarlyStopping(patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, verbose=0, save_freq="epoch")
    ])
