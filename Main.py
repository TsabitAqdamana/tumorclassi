import streamlit as st
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from PIL import Image, ImageOps

st.set_page_config(layout="wide")

st.write("""
# Web Apps - Klasifikasi Tumor Otak
Aplikasi Berbasis Web Untuk mengklasifikasi **Tumor Otak**! 
""")

if 'Brain.jpg' is not None:
    img = Image.open('Brain.jpg')
    img1 = img.resize((630, 350))
    st.image(img1, use_column_width=False)

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('model1.h5')
    return model
with st.spinner('Model sedang dijalankan..'):
    model=load_model()

st.markdown("""
<style>
.big-font {
    font-size:22px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Upload Gambar</p>', unsafe_allow_html=True)

file = st.file_uploader("Please upload a brain scan file", type=["jpg", "png", "jpeg"])

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):

    size = (100, 100)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #img_resize = (cv2.resize(img, dsize=(75, 75),   interpolation=cv2.INTER_CUBIC))/255.

    img_reshape = img[np.newaxis,...]

    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=False)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    class_name=['Tumor Otak', 'Otak Sehat']
    st.write(predictions)
    st.write(score)
    st.write(
        "Gambar ini teridentifikasi sebagai **{}** dengan tingkat keakuratan {:.2f} persen."
        .format(class_name[np.argmax(score)], 100 * np.max(score))
    )

