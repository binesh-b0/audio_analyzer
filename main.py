# Importing required libraries
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Dictionary to map emotion labels to their corresponding identifiers
emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# The emotions we are detecting
emotion_observed = ['happy', 'fearful', 'angry', 'sad']

def feature_extract(file_name, mfcc=True, chroma=True, mel=True):
    """
    Function to extract features from the audio file.
    Extracts MFCC, Chroma, and Mel Spectrogram features.
    """
    with soundfile.SoundFile(file_name) as sound:
        sound_data = sound.read(dtype="float32")
        sample_rate = sound.samplerate
        result = np.array([])

        if chroma:
            short_fourier_transform = np.abs(librosa.stft(sound_data))
        if mfcc:
            mel_freg_coef_mean = np.mean(librosa.feature.mfcc(y=sound_data,
                                                              sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mel_freg_coef_mean))
        if chroma:
            chroma_mean = np.mean(librosa.feature.chroma_stft(S=short_fourier_transform,
                                                              sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_mean))
        if mel:
            mel_spect_freq_mean = np.mean(librosa.feature.melspectrogram(y=sound_data,
                                                                         sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_spect_freq_mean))
    return result

def data_load(test_size=0.2):
    """
    Function to load data, extract features, and split into training and testing sets.
    """
    x, y = [], []
    error_count = 0
    processed_count = 0

    for file in glob.glob("G:\\PClass\\dataset\\Ravdess\\audio_speech_actors_01-24\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion_value = emotion_dict[file_name.split("-")[2]]
        if emotion_value not in emotion_observed:
            continue

        processed_count += 1
        print(f"Processing file {processed_count}: {file_name}")

        try:
            feature_values = feature_extract(file, mfcc=True, chroma=True, mel=True)
            x.append(feature_values)
            y.append(emotion_value)
        except Exception as e:
            error_count += 1
            print(f"Error processing file {file_name}: {e}")
            continue

    print(f'Total errors: {error_count}')
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_load(test_size=0.25)
    print(f"Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}")
    print(f'Features extracted: {x_train.shape[1]}')

    # Defining the MLP Classifier model and fitting the data
    model = MLPClassifier(alpha=0.01, batch_size=100, epsilon=1e-08, hidden_layer_sizes=(300,), 
                          learning_rate='adaptive', max_iter=800, verbose=True)
    model.fit(x_train, y_train)

    # Saving the trained model as a pickle string
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    y_pred = model.predict(x_test)

    # Calculating the accuracy of the model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
