import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
from keras.models import load_model


model = load_model('./Model/model-cnn.h5',compile=False)
lab = {0: 'Andaliman', 1: 'Cabe jawa', 2: 'Cengkeh', 3: 'Kapulaga', 4: 'Kayu manis', 5: 'Lada', 6: 'Pala'}

# untuk menghilangkan menu yang tidak diperlukan saja
hide_menu = """
<style>
#MainMenu { 
    visibility:hidden;
}
footer{
    visibility:visible;
}
header{
    visibility:hidden;
}
footer:after{
    content:'Copyright @ 2023: Sandi Hermawan';
    display:block;
    position:relative;
    color: tomato;
}
</style>
"""

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    img1 = Image.open('./meta/logo1.png')
    img1 = img1.resize((324,324))
    st.set_page_config(layout="wide")
    st.image(img1,use_column_width=False)
    st.title("Klasifikasi Rempah - Rempah")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>Prediksi rempah rempah masih terbatas yakni : </h4>
                    <p style='text-align: left; color: #d73b5c;'>Andaliman, Cengkeh, Cabe Jawa, Kapulaga, Kayu Manis, Lada dan Pala</p>''',
                unsafe_allow_html=True)

    st.markdown(hide_menu, unsafe_allow_html=True)

    img_file = st.file_uploader("Pilih Gambar Rempah-Rempah", type=["jpg", "png"])
    
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)

            html_str = f"""
            </style>
            <p style='text-align: left; color: #64dd17;'>Ai : Itu Adalah {result}</p>
            """
            st.markdown(html_str, unsafe_allow_html=True)
            
run()