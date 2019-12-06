from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "0.1.0"
__author__ = "Abien Fred Agarap"

from flask import flash
from flask import Flask
from flask import render_template
from flask import request
from keras.models import load_model
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def home():
    filename = "http://0.0.0.0:5000/static/" + "class.png"
    return render_template("index.html", class_finding=filename)


@app.route("/", methods=["POST"])
def classify():
    text = request.form["text"]
    assert "," in text, "FormatError: Please enter comma-separated features"

    text = text.strip(",")
    text = text.split(",")
    data = np.array(text, np.int)

    model = load_model("dnn.h5")
    prediction = np.argmax(model.predict(np.reshape(data, (-1, data.shape[0]))))

    print("Prediction : {}".format(prediction))

    class_finding = "benign.png" if prediction == 0 else "malignant.png"

    filename = "http://0.0.0.0:5000/static/" + class_finding

    return render_template("index.html", class_finding=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
