import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from main import feature_extract

app = Flask(__name__)
app.config['AUDIO_UPLOADS'] = app.root_path + "/audio/"
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    audio_data = []
    file = request.files["file"]
    user_name = request.form.get("name")
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["AUDIO_UPLOADS"], filename))
    features = feature_extract(os.path.join(app.config["AUDIO_UPLOADS"], filename), mfcc=True, chroma=True, mel=True)
    audio_data.append(features)
    prediction = model.predict(audio_data)
    return render_template("index.html", **locals())


if __name__ == "__main__":
    app.run(debug=True)