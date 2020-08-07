import numpy as np 
import pandas as pd 
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
st.set_option('deprecation.showfileUploaderEncoding', False)
import cv2
from PIL import Image, ImageOps
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = tf.keras.models.load_model('my_model.hdf5')



st.write("""
         # Plant Pathology
         """
         )

st.write("This is a image classification web app to predict Plant Disease")



file = st.file_uploader("Please upload an image file", type=["jpg", "png",'jpeg'])



def import_and_predict(image_data, model):
    
        size = (128,128)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(128, 128),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("is Healthy!")
    elif np.argmax(prediction) == 1:
        st.write("has Multiple Diseases!")
    elif np.argmax(prediction) == 2:
        st.write("has Rust!")
    else: 
        st.write("has Scab!")
    
    
    
    st.write(prediction)




