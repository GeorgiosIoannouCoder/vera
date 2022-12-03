import librosa  # Music and audio analysis.
import numpy as np  # Data wrangling.
import os  # Manipulate operating system interfaces.
import pandas as pd  # Data handling.

# Python-dotenv reads key-value pairs from a .env file and can set them as environment variables.
from dotenv import load_dotenv

load_dotenv()

from keras.utils import np_utils  # Keras utilities.

# To split data in training/validating/testing.
from sklearn.model_selection import train_test_split

# To encode target labels with value between 0 and n_classes-1.
# To perform standardization by centering and scaling.
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statistics import mode  # Find the most likely predicted emotion.
from tensorflow.keras.models import load_model  # To load the model.

model_path = os.getenv("model_path")
X_path = os.getenv("X_path")
y_path = os.getenv("y_path")

# List of emotions the model was trained on.
emotions_classes = [
    "angry",
    "calm",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# Load the model.
model = load_model(model_path)

# We have X which is data augmentation + data extraction.
X = pd.read_parquet(X_path)

# We try to predict y which is the emotion.
y = pd.read_parquet(y_path)

# Convert class vector (integers) to binary class matrix.
label = LabelEncoder()
y = y.squeeze()
y = np_utils.to_categorical(label.fit_transform(y))

# Split data into training, validation, and testing.
X_train, X_remain, y_train, y_remain = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_remain, y_remain, test_size=0.5, random_state=42
)

# Standardize / Scale data.
# Standardize features by removing the mean and scaling to unit variance.
# z = (x - u) / s
standard_scaler = StandardScaler()
X_train = standard_scaler.fit_transform(X_train.values)
X_valid = standard_scaler.transform(X_valid.values)
X_test = standard_scaler.transform(X_test.values)

# Next 4 functions are Audio Data Augmentation:
# Noise Injection.
def inject_noise(data, sampling_rate=0.035, threshold=0.075, random=False):
    if random:
        sampling_rate = np.random.random() * threshold
    noise_amplitude = sampling_rate * np.random.uniform() * np.amax(data)
    augmented_data = data + noise_amplitude * np.random.normal(size=data.shape[0])
    return augmented_data


# Pitching.
def pitching(data, sampling_rate, pitch_factor=0.7, random=False):
    if random:
        pitch_factor = np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


# Zero crossing rate.
def zero_crossing_rate(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(
        y=data, frame_length=frame_length, hop_length=hop_length
    )
    return np.squeeze(zcr)


# Root mean square.
def root_mean_square(data, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rms)


# Mel frequency cepstral coefficients.
def mel_frequency_cepstral_coefficients(
    data, sampling_rate, frame_length=2048, hop_length=512, flatten: bool = True
):
    mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


# Combined audio data feature extraction.
def feature_extraction(data, sampling_rate, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack(
        (
            result,
            zero_crossing_rate(data, frame_length, hop_length),
            root_mean_square(data, frame_length, hop_length),
            mel_frequency_cepstral_coefficients(
                data, sampling_rate, frame_length, hop_length
            ),
        )
    )
    return result


# Duration and offset act as placeholders because there is no audio in start and the ending of
# each audio file is normally below three seconds.
# Combine audio data augmentation and audio data feature extraction.
def get_features(file_path, duration=2.5, offset=0.6):
    data, sampling_rate = librosa.load(file_path, duration=duration, offset=offset)

    # No audio data augmentation.
    audio_1 = feature_extraction(data, sampling_rate)
    audio = np.array(audio_1)

    # Inject Noise.
    noise_audio = inject_noise(data, random=True)
    audio_2 = feature_extraction(noise_audio, sampling_rate)
    audio = np.vstack((audio, audio_2))

    # Pitching.
    pitch_audio = pitching(data, sampling_rate, random=True)
    audio_3 = feature_extraction(pitch_audio, sampling_rate)
    audio = np.vstack((audio, audio_3))

    # Pitching and Inject Noise.
    pitch_audio_1 = pitching(data, sampling_rate, random=True)
    pitch_noise_audio = inject_noise(pitch_audio_1, random=True)
    audio_4 = feature_extraction(pitch_noise_audio, sampling_rate)
    audio = np.vstack((audio, audio_4))

    audio_features = audio

    return audio_features


# Increase ndarray dimensions to [4,2376].
def increase_ndarray_size(features_test):
    tmp = np.zeros([4, 2377])
    offsets = [0, 1]
    insert_here = tuple(
        [
            slice(offsets[dim], offsets[dim] + features_test.shape[dim])
            for dim in range(features_test.ndim)
        ]
    )

    tmp[insert_here] = features_test
    features_test = tmp
    features_test = np.delete(features_test, 0, axis=1)

    return features_test


# Determine if ndarray needs to be increase in size.
def increase_array_size(audio_features):
    if audio_features.shape[1] < 2376:
        audio_features = increase_ndarray_size(audio_features)
    return audio_features


# Make the prediction.
def predict(audio_features):
    audio_features = standard_scaler.transform(audio_features)
    audio_features = np.expand_dims(audio_features, axis=2)

    y_pred = model.predict(audio_features)
    y_pred = np.argmax(y_pred, axis=1)

    try:
        # Model debugging.
        # print("\nPredicted emotion for each and every feature extraction.\n\n", y_pred)
        # print("\nAvailable emotions_classes = ", emotions_classes)
        # print("\nModel predicted emotion: ", emotions_classes[mode(y_pred)])
        return emotions_classes[mode(y_pred)]
    except:
        return emotions_classes[y_pred[0]]
