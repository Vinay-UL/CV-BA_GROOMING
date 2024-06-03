
import cv2
import time
import math
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
#from azure.storage.blob.blockblobservice import BlockBlobService
import json
import base64
import logging
from io import BytesIO
import tensorflow as tf
from tensorflow import keras
import os



def load_uniform_classification_model():
    global uniformModel
    global img_height
    global img_width
    global uniform_class_names
    print("function is in load uniform classification")
    uniformModel = tf.keras.models.load_model('./model/efficientNetB7_baseFreezed650_Sigmoid')
    img_height = 180
    img_width = 180
    uniform_class_names = ['no_uniform', 'uniform']



def getPrediction(imagePath):
    img_height = 180
    img_width = 180
    uniform_class_names = ['no_uniform', 'uniform']


    img = keras.preprocessing.image.load_img(
        imagePath, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = uniformModel.predict(img_array)

    className = uniform_class_names[int(round(predictions[0][0]))]

    return className


def init():

    logging.info('loading the model')
    try:
        load_uniform_classification_model()

        print("models loaded successfully")
    except Exception as e:
        logging.error(e)
        

 


def run(raw_data):
    try:
        print('runmethod')
        start_time = time.time()
        
        #preds = pred_image(bytes(image, encoding='utf8'))
        
        
        time_taken = (time.time() - start_time)
        image = json.loads(raw_data)["image"]
        encoded_img=bytes(image, encoding='utf8')
        image = np.array(Image.open(BytesIO(base64.b64decode(encoded_img))))
        uniformImagePath = "./temp.png"
        cv2.imwrite(uniformImagePath,image)
        className = getPrediction(uniformImagePath)
        os.remove(uniformImagePath)
        prediction = "No"
        if className == 'uniform':
           prediction = "Yes"
        time_taken = (time.time() - start_time)
        response = {"predictions": str(prediction),"Processing_time": str(time_taken)}
        return response
    except Exception as e:
        error = str(e)
        return error
    
