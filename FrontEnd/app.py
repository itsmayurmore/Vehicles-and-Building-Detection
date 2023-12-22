from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ScratchModel import NeuralNet, image_loader
 
app = Flask(__name__)

mark2 = torch.hub.load('ultralytics/yolov5','custom',path='model/mark2.pt',force_reload=False,verbose=False)

mark1 = NeuralNet(num_classes=4)
mark1.load_state_dict(torch.load('model/mark1.model'))
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "itsmayurmore"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/predict', methods=['POST'])
def predict():
    if os.path.exists('static/saved/plot.png'):
        os.remove('static/saved/plot.png')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('')
    plt.imshow(np.squeeze(result.render()))
    plt.savefig('static/saved/plot.png')

    #Mark 1
    if os.path.exists('static/saved/plot1.png'):
        os.remove('static/saved/plot1.png')
    classess = ['2-wheeler', '4-wheeler', 'Bicycle', 'Buildings']
    prediction = mark1(imageName).detach().numpy()[0]
    index = np.where(prediction>=prediction.max())[0][0]

    fig  = plt.figure()
    img=mpimg.imread(imageFileName)
    ax = fig.add_subplot(1,1,1)
    ax.set_title(classess[index])
    plt.imshow(np.squeeze(img))
    plt.savefig('static/saved/plot1.png')                                                   
  
    return render_template('predict.html',file1='static/saved/plot1.png',file2='static/saved/plot.png',cls1 = classess[index],cls2 = result)
     

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part' , category='warning')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading' , category='warning')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below',category='info')
        global imageFileName , result , imageName
        imageFileName = UPLOAD_FOLDER+filename
         
        result = mark2(UPLOAD_FOLDER+filename)
         
        imageName = image_loader(UPLOAD_FOLDER+filename)
        
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg only' , category='warning')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(debug=False)