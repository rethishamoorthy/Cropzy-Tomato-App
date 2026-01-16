# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image

# app = Flask(__name__)

# model = tf.keras.models.load_model("tomato_disease_model.keras")

# class_names = [
#     "Tomato___Early_blight",
#     "Tomato___Late_blight",
#     "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
#     "Tomato___Tomato_mosaic_virus",
#     "Tomato___healthy"
# ]

# @app.route("/predict", methods=["POST"])
# def predict():
#     file = request.files["file"]
#     img = image.load_img(file, target_size=(224,224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     pred = model.predict(img_array)
#     result = class_names[np.argmax(pred)]

#     return jsonify({"prediction": result})

# if __name__ == "__main__":
#     app.run(debug=True)



import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("tomato_disease_model.keras")

class_names = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# App UI
st.title("🍅 Cropzy - Tomato Disease Detection")
st.write("Upload a tomato leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    st.success(f"Prediction: {class_names[np.argmax(pred)]}")
