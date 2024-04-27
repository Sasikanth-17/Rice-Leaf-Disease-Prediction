from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = load_model('rice_leaf_disease_model_updated.h5')

# Dictionary to map class indices to class names
class_labels = {0: 'Bacterial leaf blight', 1: 'Blast', 2: 'Brown spot', 3: 'Leaf smut', 4: 'Tungro'}


# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the classification request
@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get the image file from the request
        img_file = request.files['image']
        
        # Save the image to a temporary file
        temp_img_path = 'temp_img.jpg'
        img_file.save(temp_img_path)
        
        # Preprocess the image
        img_array = preprocess_image(temp_img_path)
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(prediction)
        
        # Get the predicted class label
        predicted_class_label = class_labels[predicted_class_index]

        # Render the result page with the predicted class label
        return render_template('result.html', class_label=predicted_class_label)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
