# Augmentation + save augmented images under augmented folder

from keras.preprocessing.image import ImageDataGenerator  
import os
from pathlib import Path


IMAGE_SIZE = 224
BATCH_SIZE = 10
NO_OF_IMAGES = 25
SOURCE_DIR = "/home/daj/Desktop/datasets/sample"
LABELS = os.listdir(SOURCE_DIR)

datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.2, 1.3),
        fill_mode='constant', cval=125)

for label in LABELS:
    # datagen_kwargs = dict(rescale=1./255)  
    dataflow_kwargs = dict(target_size=(IMAGE_SIZE, IMAGE_SIZE), 
                            batch_size=BATCH_SIZE, interpolation="bilinear")
    
    target = Path(r"/home/daj/Desktop/datasets/sample/augmented") / label
    target.mkdir(parents=True, exist_ok=True)

    i = 1
    for batch in datagen.flow_from_directory(SOURCE_DIR,
                                            save_to_dir= target , 
                                            save_prefix='aug', 
                                            classes=[label], 
                                            **dataflow_kwargs):
        
        i += 1
        if i > NO_OF_IMAGES:
            break