import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

emotion_observed = ['happy', 'fearful', 'angry', 'sad']


def feature_extract(file_name, mfcc, chroma, mel):
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
            mel_spect_freq_mean = np.mean(librosa.feature.melspectrogram(sound_data,
                                                                         sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_spect_freq_mean))
    return result


def data_load(test_size=0.2):
    x, y = [], []
    i=0
    j=0
    for file in glob.glob("G:\\PClass\\dataset\\Ravdess\\audio_speech_actors_01-24\\Actor_*\\*.wav"):

        file_name = os.path.basename(file)
        global emotion_dict
        global emotion_observed
        emotion_value = emotion_dict[file_name.split("-")[2]]
        if emotion_value not in emotion_observed:
            continue
        print(i)
        i+=1
        try:
            feature_values = feature_extract(file, mfcc=True, chroma=True, mel=True)
        except:
            j+=1
            continue

        x.append(feature_values)
        y.append(emotion_value)
    print('error',j)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = data_load(test_size=0.25)
    print((x_train.shape[0], x_test.shape[0]))
    print(f'Features extracted: {x_train.shape[1]}')

    model = MLPClassifier(alpha=0.01, batch_size=100, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=800,verbose=True)
    model.fit(x_train, y_train)

    pickle.dump(model, open('model.pkl', 'wb'))

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy*100))



