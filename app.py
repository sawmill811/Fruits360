from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

# Imports
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import bz2

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

from sklearn.ensemble import RandomForestClassifier
import cv2
import matplotlib.pyplot as plt
import pickle
import _pickle as cPickle
import numpy as np
from PIL import Image
import blosc
import keras


app = Flask(__name__)
app.config['IMG_SIZE'] = (96,96)

# Load the model
model = keras.models.load_model("static/models/vgg16_96x96/content/vgg16_96x96")

# Landing Page
@app.route('/', methods=['GET','POST'])
def send_form():
    return render_template('main.html', data="No image uploaded!")

# Prediction Page
@app.route('/prediction', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/' to submit form"
    if request.method == 'POST':
        form_data = request.files

    return render_template('main2.html', data = predict2(form_data), predImage="static/uploads/imgPred.jpg")


def load_image(path, img_size):     # load image from provided path and convert it to rgb and rescale it to 1./255
  img = Image.open(path)
  img = img.convert('RGB')
  img = img.resize(img_size)
  img.save("static/uploads/imgPred.jpg")
  img = np.array(img)
  img = img / 255.
  img = img[np.newaxis, :]
  return img

# Get the prediction
def get_class(index):
    classes = ["Apple Braeburn", "Apple Crimson Snow", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith", "Apple Pink Lady", "Apple Red 1", "Apple Red 2"
    , "Apple Red 3",'Apple Red Delicious','Apple Red Yellow 1','Apple Red Yellow 2','Apricot','Avocado','Avocado ripe','Banana','Banana Lady Finger','Banana Red','Beetroot'
    ,'Blueberry','Cactus fruit','Cantaloupe 1','Cantaloupe 2','Carambula','Cauliflower','Cherry 1','Cherry 2','Cherry Rainier','Cherry Wax Black','Cherry Wax Red'
    ,'Cherry Wax Yellow','Chestnut','Clementine','Cocos','Corn','Corn Husk','Cucumber Ripe','Cucumber Ripe 2','Dates','Eggplant','Fig','Ginger Root','Granadilla','Grape Blue'
    ,'Grape Pink','Grape White','Grape White 2','Grape White 3','Grape White 4','Grapefruit Pink','Grapefruit White','Guava','Hazelnut','Huckleberry','Kaki','Kiwi','Kohlrabi'
    ,'Kumquats','Lemon','Lemon Meyer','Limes','Lychee','Mandarine','Mango','Mango Red','Mangostan','Maracuja','Melon Piel de Sapo','Mulberry','Nectarine','Nectarine Flat'
    ,'Nut Forest','Nut Pecan','Onion Red','Onion Red Peeled','Onion White','Orange','Papaya','Passion Fruit','Peach','Peach 2','Peach Flat','Pear','Pear 2','Pear Abate','Pear Forelle'
    ,'Pear Kaiser','Pear Monster','Pear Red','Pear Stone','Pear Williams','Pepino','Pepper Green','Pepper Orange','Pepper Red','Pepper Yellow','Physalis','Physalis with Husk'
    ,'Pineapple','Pineapple Mini','Pitahaya Red','Plum','Plum 2','Plum 3','Pomegranate','Pomelo Sweetie','Potato Red','Potato Red Washed','Potato Sweet','Potato White'
    ,'Quince','Rambutan','Raspberry','Redcurrant','Salak','Strawberry','Strawberry Wedge','Tamarillo','Tangelo','Tomato 1','Tomato 2','Tomato 3','Tomato 4','Tomato Cherry Red'
    ,'Tomato Heart','Tomato Maroon',"Tomato not Ripened","Tomato Yellow","Walnut","Watermelon"]
    return classes[index]

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clearFiles():
    dir = "static/uploads/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def predict2(form_data):

    clearFiles()
    file = form_data['img']
    filename = secure_filename(file.filename)

    if file and allowed_file(filename):
        # Save the file to the uploads folder
        file.save(os.path.join("static/uploads", filename))

        

        # Load the image
        img_path = os.path.join("static/uploads", filename)
        img = load_image(img_path, app.config['IMG_SIZE'])

        # Predict
        y_pred = model.predict(img)
        y_classes_pred = y_pred.argmax(axis=-1)
        y_class_labels = get_class(y_classes_pred.item())
        label = y_class_labels

        # Return the prediction
        return label
    else:
        return "Incorrect file format"
