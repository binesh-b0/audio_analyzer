import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from main import feature_extract

app = Flask(__name__)
app.config['AUDIO_UPLOADS'] = app.root_path + "/audio/"
model = pickle.load(open("model.pkl", "rb"))
ALLOWED_EXTENSIONS = {'wav','mp3'}

def check_filetypes(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    audio_data = []
    file = request.files["file"]
    user_name = request.form.get("name")
    filename = secure_filename(file.filename)
    if not check_filetypes(file.filename):
        return render_template("index.html",error = "Not an audio file")
    file.save(os.path.join(app.config["AUDIO_UPLOADS"], filename))
    try:
        features = feature_extract(os.path.join(app.config["AUDIO_UPLOADS"], filename), mfcc=True, chroma=True, mel=True)
        audio_data.append(features)
        prediction = model.predict(audio_data)
        return render_template("index.html", **locals())
    except:
        return render_template("index.html", error = "Unsupported audio" )



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)