
import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random 
import keras_ocr
import imgaug
import sklearn.model_selection
import string

# FINE TUNING RECOGNIZER MODEL WITH SYNTHETIC DATA 

device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

tf.debugging.set_log_device_placement(True)


#Detecting GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Path to the data directory
data_dir = Path("/home/alan/Documents/TextRecognitionDataGenerator/trdg/out/")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [(img.split(os.path.sep)[-1].split(".jpg")[0]).split("_", 1) [0] for img in images]
characters = set(char for label in labels for char in label)
N = len(images)
box = [None] * N
#get ist of tuples of paths of images, box and text label
data = list(zip (images, box, labels))
# make lower text of labels
data = [(filepath, box, word.lower()) for filepath, box, word in data]

print("Number of examples: ", N)
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)


# Spliting data 

ixRand  = list(range(N))
random.shuffle(ixRand)
train_data = [data[e] for e in ixRand[:round(N*.8)]]
test_data = [data[e] for e in ixRand[round(N*.8):]]

#Building the model

recognizer = keras_ocr.recognition.Recognizer()
recognizer.compile()

#make alphabet spanish
alphabet = ''.join(sorted (set(  'ñ!?¡-¿' + recognizer.alphabet)))

# building keras dataset generator

batch_size = 8
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0)),
])


train_labels, validation_labels = sklearn.model_selection.train_test_split(train_data, test_size=0.2, random_state=42)
(training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
    (
        keras_ocr.datasets.get_recognizer_image_generator(
            labels=labels,
            height=recognizer.model.input_shape[1],
            width=recognizer.model.input_shape[2],
            alphabet=alphabet,
            augmenter=augmenter
        ),
        len(labels) // batch_size
    ) for labels, augmenter in [(train_labels, augmenter), (validation_labels, None)]     
]
training_gen, validation_gen = [
    recognizer.get_batch_generator(
        image_generator=image_generator,
        batch_size=batch_size
    )
    for image_generator in [training_image_gen, validation_image_gen]
]


image, text = next(training_image_gen)
print('text:', text)
_ = plt.imshow(image)



#%%Training recognizer model

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=False),
    tf.keras.callbacks.ModelCheckpoint('recognizer_synthetic.h5', monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.CSVLogger('recognizer_synthetic.csv')
]
recognizer.training_model.fit_generator(
    generator=training_gen,
    steps_per_epoch=training_steps,
    validation_steps=validation_steps,
    validation_data=validation_gen,
    callbacks=callbacks,
    epochs=1000,
)


#%% test inference of recognizer model


image_filepath, _, actual = test_data[1]
predicted = recognizer.recognize(image_filepath)
a = (f'Predicted: {predicted}, Actual: {actual}')
_ = plt.imshow(keras_ocr.tools.read(image_filepath))



#%%

# FINE TUNING DETECTOR MODEL WITH ICDAR DATASET

dataset = keras_ocr.datasets.get_icdar_2013_detector_dataset(
    cache_dir='.',
    skip_illegible=False
)

#split dataset train and test


train, validation = sklearn.model_selection.train_test_split(
    dataset, train_size=0.8, random_state=42
)
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.Affine(
    scale=(1.0, 1.2),
    rotate=(-5, 5)
    ),
    imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
])
generator_kwargs = {'width': 640, 'height': 640}
training_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=train,
    augmenter=augmenter,
    **generator_kwargs
)
validation_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=validation,
    **generator_kwargs
)

#Build and Train model

detector = keras_ocr.detection.Detector()

batch_size = 1
training_generator, validation_generator = [
    detector.get_batch_generator(
        image_generator=image_generator, batch_size=batch_size
    ) for image_generator in
    [training_image_generator, validation_image_generator]
]
detector.model.fit_generator(
    generator=training_generator,
    steps_per_epoch=math.ceil(len(train) / batch_size),
    epochs=1000,
    workers=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
        tf.keras.callbacks.CSVLogger(os.path.join(data_dir, 'detector_icdar2013.csv')),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(data_dir, 'detector_icdar2013.h5'))
    ],
    validation_data=validation_generator,
    validation_steps=math.ceil(len(validation) / batch_size)
)
