import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tqdm import tqdm
from azureml.core.run import Run
run = Run.get_submitted_run()

img_height=180
img_width=180
batch_size=8

mounted_input_path = sys.argv[1]
print(os.listdir(mounted_input_path))
data_dir = os.path.join(mounted_input_path)

class_names = ['no_uniform','uniform']
model=tf.keras.models.load_model('model/efficientNetB7_baseFreezed650_Sigmoid')

uniformImagePath=data_dir+'/'
img_height=180
img_width=180
uniformCounter = 0
for imageName in tqdm(os.listdir(data_dir)):
    img=keras.preprocessing.image.load_img(
        uniformImagePath+imageName, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)

    
    className = class_names[int(round(predictions[0][0]))]
    if className == 'with_uniform':
        uniformCounter += 1
    else: 
        print(
            "This image {} most likely belongs to {}"
            .format(imageName, className)
        )
print(uniformCounter)