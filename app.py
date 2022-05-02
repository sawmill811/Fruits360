from __future__ import absolute_import, division, print_function, unicode_literals

# Imports
import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

import numpy as np
from PIL import Image
import keras


app = Flask(__name__)
app.config['IMG_SIZE'] = (96,96)

# Pipeline class

class Pipeline:

  def __predict_single_img_class(self, path, model_object):                  # important for the backend to function smoothly
    image_x = self.__load_image(path)
    y_pred = model_object.predict(image_x)
    y_classes_pred = y_pred.argmax(axis=-1)
    y_class_labels = np.array(sorted(self.__folders))[y_classes_pred]
    label = y_class_labels[0]
    return label
  
  def __load_image(self, path):                                # load image from provided path and convert it to rgb and rescale it to 1./255
    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize(self.__image_dimensions_no_channels)
    img.save("static/uploads/imgPred.jpg")
    img = np.array(img)
    img = img / 255.
    img = img[np.newaxis, :]
    return img

  def __load_model(self, model_name):
    # loads the model from the zip file my_model.zip in the current working directory
    self.__model = keras.models.load_model(f"my_model/content/{model_name}")

  def summary(self):
    return self.__model.summary()

  def __init__(self):
    self.__image_dimensions_no_channels = (96, 96)
    self.__folders = sorted(["Apple Braeburn", "Apple Crimson Snow", "Apple Golden 1", "Apple Golden 2", "Apple Golden 3", "Apple Granny Smith", "Apple Pink Lady", "Apple Red 1", "Apple Red 2", "Apple Red 3",'Apple Red Delicious','Apple Red Yellow 1','Apple Red Yellow 2','Apricot','Avocado','Avocado ripe','Banana','Banana Lady Finger','Banana Red','Beetroot','Blueberry','Cactus fruit','Cantaloupe 1','Cantaloupe 2','Carambula','Cauliflower','Cherry 1','Cherry 2','Cherry Rainier','Cherry Wax Black','Cherry Wax Red','Cherry Wax Yellow','Chestnut','Clementine','Cocos','Corn','Corn Husk','Cucumber Ripe','Cucumber Ripe 2','Dates','Eggplant','Fig','Ginger Root','Granadilla','Grape Blue','Grape Pink','Grape White','Grape White 2','Grape White 3','Grape White 4','Grapefruit Pink','Grapefruit White','Guava','Hazelnut','Huckleberry','Kaki','Kiwi','Kohlrabi','Kumquats','Lemon','Lemon Meyer','Limes','Lychee','Mandarine','Mango','Mango Red','Mangostan','Maracuja','Melon Piel de Sapo','Mulberry','Nectarine','Nectarine Flat','Nut Forest','Nut Pecan','Onion Red','Onion Red Peeled','Onion White','Orange','Papaya','Passion Fruit','Peach','Peach 2','Peach Flat','Pear','Pear 2','Pear Abate','Pear Forelle','Pear Kaiser','Pear Monster','Pear Red','Pear Stone','Pear Williams','Pepino','Pepper Green','Pepper Orange','Pepper Red','Pepper Yellow','Physalis','Physalis with Husk','Pineapple','Pineapple Mini','Pitahaya Red','Plum','Plum 2','Plum 3','Pomegranate','Pomelo Sweetie','Potato Red','Potato Red Washed','Potato Sweet','Potato White','Quince','Rambutan','Raspberry','Redcurrant','Salak','Strawberry','Strawberry Wedge','Tamarillo','Tangelo','Tomato 1','Tomato 2','Tomato 3','Tomato 4','Tomato Cherry Red','Tomato Heart','Tomato Maroon',"Tomato not Ripened","Tomato Yellow","Walnut","Watermelon"])
    self.__load_model('vgg16_96x96')
  
  def predict(self, path_img):
    return self.__predict_single_img_class(path_img, self.__model)

# Load the model
pipe = Pipeline()

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

    return render_template('main2.html', data = predict(form_data), predImage="static/uploads/imgPred.jpg")

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clearFiles():
    dir = "static/uploads/"
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

def predict(form_data):

    clearFiles()
    file = form_data['img']
    filename = secure_filename(file.filename)

    if file and allowed_file(filename):
        # Save the file to the uploads folder
        file.save(os.path.join("static/uploads", filename))

        # Image Path
        img_path = os.path.join("static/uploads", filename)

        # Return the prediction from the Pipeline
        return pipe.predict(img_path)
    else:
        return "Incorrect file format"
