import os
import json
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import plate_locator

app = Flask(__name__)
CORS(app)  # 解决跨域问题

def get_prediction(img_path):
    try:
        text = [plate_locator.predict_muban(img_path)]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info

def get_prediction_dl(img_path):
    try:
        text = [plate_locator.resnet_predict(img_path)]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info

@app.route("/predict", methods=["POST"])
def predict():
    image_path = "./images/"+request.files["file"].filename
    request.files["file"].save(image_path)
    print(image_path)
    info = get_prediction(image_path)
    return jsonify(info)

@app.route("/predict_dl", methods=["POST"])
def predict_dl():
    image_path = "./images/"+request.files["file"].filename
    request.files["file"].save(image_path)
    print(image_path)
    info = get_prediction_dl(image_path)
    return jsonify(info)

@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)




