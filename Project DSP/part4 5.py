import librosa
import librosa.display
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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

# Paths to the labeled "yes" and "no" speech samples
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
mfcc_yes1 = extract_mfcc_features(audio1)
mfcc_yes2 = extract_mfcc_features(audio2)
mfcc_yes3 = extract_mfcc_features(audio5)
mfcc_yes4 = extract_mfcc_features(audio7)
mfcc_yes5 = extract_mfcc_features(audio8)
mfcc_no1 = extract_mfcc_features(audio3)
mfcc_no2 = extract_mfcc_features(audio4)
mfcc_no3 = extract_mfcc_features(audio6)
mfcc_no4 = extract_mfcc_features(audio9)
mfcc_no5 = extract_mfcc_features(audio10)

# Define the training data (labeled samples)
X_train = [mfcc_yes1, mfcc_yes2, mfcc_yes3, mfcc_yes4, mfcc_yes5, mfcc_no1, mfcc_no2, mfcc_no3, mfcc_no4, mfcc_no5]  # Features of the labeled samples
y_train = ['yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no', 'no', 'no']  # Corresponding labels ("yes" or "no")

# Encode the labels into numerical values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Split the data into training and validation sets
X_train, X_val, y_train_encoded, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Define the classifier model
model = RandomForestClassifier()

# Perform grid search with stratified k-fold cross-validation
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10]
}
cv = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(model, param_grid, cv=cv)
grid_search.fit(X_train, y_train_encoded)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model on the validation set
y_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_val, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=1)

print("Validation Accuracy:", accuracy)
print("Validation Precision:", precision)
print("Validation Recall:", recall)
print("Validation F1 Score:", f1)

# Define the function to predict the category of new speech samples
def predict_category(audio_path):
    # Extract MFCC features from the new speech sample
    new_features = extract_mfcc_features(audio_path)

    # Reshape the features if necessary (assuming a single sample)
    new_features_reshaped = new_features.reshape(1, -1)

    # Pass the reshaped features through the classifier
    predicted_category = best_model.predict(new_features_reshaped)[0]

    # Convert the predicted category back to the original label
    label = label_encoder.inverse_transform([predicted_category])[0]

    return label

# Example usage:
new_audio_path = "C:/Users/Project DSP/my-dsp-recording/dsp recording/no5.wav" # Replace with the actual path to the new audio sample
predicted_label = predict_category(new_audio_path)
print("Predicted label:", predicted_label)
