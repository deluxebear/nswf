import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import predict
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = set([ 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = predict.load_model('./nsfw.299x299.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/nsfw', methods=[ 'POST'])
def upload_file():
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename)
            res= predict.classify(model,filename)
            os.remove(filename)
            return res
        return {"message","error"}