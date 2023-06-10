import librosa
import librosa.display
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier,SGDRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
# Define the function to extract MFCC features from audio samples
def extract_mfcc_features(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)  # Load the audio sample
    stft = librosa.stft(audio)
    spectrogram = np.abs(stft)
    power_spectrogram = spectrogram ** 2  # Compute power spectrogram
    mel_spec = librosa.feature.melspectrogram(S=power_spectrogram, sr=sr)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13)  # Extract MFCC features
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

# Paths to the labeled "co" and "khong" speech samples
audio1 = "C:/Users/Project DSP/yes.wav"
audio2 = "C:/Users/Project DSP/yess.wav"
audio3 = "C:/Users/Project DSP/no.wav"
audio4 = "C:/Users/Project DSP/noo.wav"
audio5 = "C:/Users/Project DSP/my-dsp-recording/dsp recording/yes1.wav"
audio6 = "C:/Users/Project DSP/my-dsp-recording/dsp recording/no1.wav"
audio7 = "C:/Users/Project DSP/yesss.wav"
audio8 = "C:/Users/Project DSP/yessss.wav"
audio9 = "C:/Users/Project DSP/nooo.wav"
audio10 = "C:/Users/Project DSP/noooo.wav"
# Extract MFCC features for each labeled sample and create the corresponding labels
mfcc_co1 = extract_mfcc_features(audio1)
mfcc_co2 = extract_mfcc_features(audio2)
mfcc_co3 = extract_mfcc_features(audio5)
mfcc_co4 = extract_mfcc_features(audio7)
mfcc_co5 = extract_mfcc_features(audio8)
mfcc_khong1 = extract_mfcc_features(audio3)
mfcc_khong2 = extract_mfcc_features(audio4)
mfcc_khong3 = extract_mfcc_features(audio6)
mfcc_khong4 = extract_mfcc_features(audio9)
mfcc_khong5 = extract_mfcc_features(audio10)
# Define the training data (labeled samples)
X_train = [mfcc_co1, mfcc_co2, mfcc_co3, mfcc_co4, mfcc_co5, mfcc_khong1, mfcc_khong2, mfcc_khong3, mfcc_khong4, mfcc_khong5]  # Features of the labeled samples
y_train = ['yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no']  # Corresponding labels ("yes" or "no")

# Encode the labels into numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Train the SGDClassifier model
model = SGDClassifier()
model.fit(X_train, y_train_encoded)

# Define the function to predict the category of new speech samples
def predict_category(audio_path):
    # Extract MFCC features from the new speech sample
    new_features = extract_mfcc_features(audio_path)

    # Reshape the features if necessary (assuming a single sample)
    new_features_reshaped = new_features.reshape(1, -1)

    # Pass the reshaped features through the classifier
    predicted_category = model.predict(new_features_reshaped)[0]

    # Convert the predicted category back to the original label
    label = label_encoder.inverse_transform([predicted_category])[0]

    return label

# Example usage:
new_audio_path = "C:/Users/Project DSP/my-dsp-recording/dsp recording/no5.wav" # Replace with the actual path to the new audio sample
predicted_label = predict_category(new_audio_path)
print("Predicted label:", predicted_label)