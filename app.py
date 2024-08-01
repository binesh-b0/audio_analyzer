import pickle
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from main import feature_extract

# The instance of the imported Flask class will be the web application
app = Flask(__name__)
app.config['AUDIO_UPLOADS'] = os.path.join(app.root_path, "audio/")
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

# Load the saved model using pickle
model = pickle.load(open("model.pkl", "rb"))

def check_filetypes(filename):
    """
    Function to check if the uploaded file is an allowed audio type.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def main():
    """
    Default route to render the main page.
    """
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Route to handle audio file upload and prediction.
    """
    audio_data = []
    file = request.files["file"]
    user_name = request.form.get("name")
    filename = secure_filename(file.filename)
    
    # Check if the file is an audio file
    if not check_filetypes(file.filename):
        return render_template("index.html", error="Not an audio file")
    
    # Ensure the AUDIO_UPLOADS directory exists
    if not os.path.exists(app.config['AUDIO_UPLOADS']):
        os.makedirs(app.config['AUDIO_UPLOADS'])
    
    # Save the audio file
    file.save(os.path.join(app.config["AUDIO_UPLOADS"], filename))
    
    try:
        # Extract features from the audio file
        features = feature_extract(os.path.join(app.config["AUDIO_UPLOADS"], filename), mfcc=True, chroma=True, mel=True)
        audio_data.append(features)
        
        # Make prediction using the pickled model
        prediction = model.predict(audio_data)
        return render_template("index.html", prediction=prediction[0], name=user_name)
    except Exception as e:
        # Handle exceptions during feature extraction or prediction
        print(f"Error: {e}")
        return render_template("index.html", error="Unsupported audio")

# Main driver function
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Run the application
    app.run(host='0.0.0.0', port=port)
