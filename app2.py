from flask import Flask, render_template, request
from PIL import Image
import io
import base64

import numpy as np
from tensorflow import keras 
import numpy as np

modelo_cargado = keras.models.load_model('modeloCNNInception.h5')

def normalize_images_test(images):
    normalized_images = []
    for image in images:
        normalized_image = image / np.max(image)
        normalized_images.append(normalized_image)
    return normalized_images

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/modelo', methods=['POST'])
def modelo():
    image = request.files['image']

    # Leer la imagen
    img = Image.open(image)

    #Clasificar Imagen
    imagen1=[]
    imagen1.append(np.array(img.convert('I').resize((300,300)).copy()))
    #print(imagen1[0])
    imagen1normalized=normalize_images_test(imagen1)
    print(modelo_cargado.predict(np.array(imagen1normalized)))
    predict = np.where(modelo_cargado.predict(np.array(imagen1normalized)) >= 0.5, 'Fracturado','No Fracturado')
    mensaje = predict[0][0]
    print(mensaje)

    # Redimensionar la imagen manteniendo la proporción
    max_size = (500, 500)  # Tamaño máximo permitido
    img.thumbnail(max_size)

    # Convertir la imagen a base64
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return render_template('index.html', image=encoded_img, clasificacion=mensaje)

if __name__ == '__main__':
    app.run()