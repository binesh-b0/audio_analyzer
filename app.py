# importing the required libraries
import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from main import feature_extract

# The instance of imported Flask class will be the web application
app = Flask(__name__)
app.config['AUDIO_UPLOADS'] = app.root_path + "/audio/"
# Python pickle module is used for serializing and de-serializing a Python object structure
# load the saved model
model = pickle.load(open("model.pkl", "rb"))
ALLOWED_EXTENSIONS = {'wav','mp3'}

def check_filetypes(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#  the route() tells Flask what URL should trigger the corresponding  function.
@app.route("/")
def main():
    # render index.html as the default page
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    audio_data = []
    file = request.files["file"]
    user_name = request.form.get("name")
    filename = secure_filename(file.filename)
    # check if the files are audio flies
    if not check_filetypes(file.filename):
        return render_template("index.html",error = "Not an audio file")
    # save audio files
    file.save(os.path.join(app.config["AUDIO_UPLOADS"], filename))
    try:
        features = feature_extract(os.path.join(app.config["AUDIO_UPLOADS"], filename), mfcc=True, chroma=True, mel=True)
        audio_data.append(features)
        # make prediction using the pickled model
        prediction = model.predict(audio_data)
        return render_template("index.html", **locals())
    except:
        return render_template("index.html", error = "Unsupported audio" )


# main driver function
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # run() method of Flask class runs the application
    app.run(host='0.0.0.0', port=port)