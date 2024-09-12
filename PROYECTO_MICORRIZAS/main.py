from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Cargar el modelo entrenado
model_path = os.path.join('C:\\Users\\karli\\Downloads\\PROYECTO_MICORRIZAS\\PROYECTO_MICORRIZAS', 'modelo_micorriza.h5')

try:
    model = load_model(model_path)  # Intenta cargar el modelo
    print("Modelo cargado correctamente.")
except Exception as e:
    model = None  # Asegura que la variable model esté definida aunque falle la carga
    print("Error al cargar el modelo:", str(e))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Error: El modelo no se cargó correctamente."

    # Obtener la imagen del formulario
    uploaded_file = request.files['image']

    # Guardar la imagen en el servidor
    image_path = 'uploaded_image.jpg'
    uploaded_file.save(image_path)

    CONFIDENCE_THRESHOLD = 0.90  # Puedes ajustar este valor según las necesidades

    # Cargar y preprocesar la imagen
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizar

    # Hacer la predicción
    try:
        prediction = model.predict(img_array)
        # Obtener el índice de la clase predicha
        predicted_index = np.argmax(prediction[0])
        confidence = np.max(prediction[0])  # Confianza de la predicción
        # Mapear los índices a subtipos de endomicorriza
        class_names = ["Arbuscular", "Diversisporal", "Ectendomicorriza", "Gigasporal", "Glomeromycote", "Rizoctonia"]
        
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class= "La imagen no pertenece a ninguna clase conocida."
        else:
             predicted_class = f"El subtipo de endomicorriza es: {class_names[predicted_index]} con una confianza de {confidence*100:.2f}%"
    except Exception as e:
        return f"Error durante la predicción: {str(e)}"

    return f', {predicted_class}'

if __name__ == '__main__':
    app.run(debug=True)
