import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

class MusicGenreClassifier:
    def __init__(self):
        self.tabular_model = None
        self.cnn_model = None
        self.transfer_model = None

    def load_tabular_data(self, file_path=r'C:\Users\Veclar\Desktop\Final Tasks\Elevvo\Task 6\features_30_sec.csv'):
        try:
            data = pd.read_csv(file_path)
            print("Tabular data loaded successfully!")
            return data
        except Exception as e:
            print(f"Error loading tabular data: {e}")
            return None

    def train_tabular_model(self, X_train, X_test, y_train, y_test):
        try:
            self.tabular_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.tabular_model.fit(X_train, y_train)
            predictions = self.tabular_model.predict(X_test)
            print("Random Forest Classifier Report:")
            print(classification_report(y_test, predictions))
            sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d')
            plt.title("Confusion Matrix - Tabular Data")
            plt.show()
        except Exception as e:
            print(f"Error training tabular model: {e}")

    def create_spectrograms(self, audio_dir='audio', output_dir='spectrograms'):
        try:
            genres = os.listdir(audio_dir)
            os.makedirs(output_dir, exist_ok=True)
            for genre in genres:
                genre_dir = os.path.join(audio_dir, genre)
                save_dir = os.path.join(output_dir, genre)
                os.makedirs(save_dir, exist_ok=True)
                for filename in os.listdir(genre_dir):
                    file_path = os.path.join(genre_dir, filename)
                    y, sr = librosa.load(file_path, duration=30)
                    S = librosa.feature.melspectrogram(y=y, sr=sr)
                    S_DB = librosa.power_to_db(S, ref=np.max)
                    plt.figure(figsize=(2.24, 2.24))
                    librosa.display.specshow(S_DB, sr=sr)
                    plt.axis('off')
                    plt.tight_layout()
                    save_path = os.path.join(save_dir, filename.replace('.wav', '.jpg'))
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
            print("Spectrograms created successfully!")
        except Exception as e:
            print(f"Error creating spectrograms: {e}")

    def train_cnn_model(self, image_dir=None, img_size=(64, 64), batch_size=32):
        try:
            if image_dir is None:
                image_dir = os.path.join(os.path.dirname(__file__), 'spectrograms')
            print(f"Loading images from: {image_dir}")
            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            train_gen = datagen.flow_from_directory(
                image_dir, target_size=img_size, batch_size=batch_size,
                class_mode='categorical', subset='training'
            )
            val_gen = datagen.flow_from_directory(
                image_dir, target_size=img_size, batch_size=batch_size,
                class_mode='categorical', subset='validation'
            )
            self.cnn_model = Sequential([
                Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
                MaxPooling2D(2,2),
                Conv2D(64, (3,3), activation='relu'),
                MaxPooling2D(2,2),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(train_gen.num_classes, activation='softmax')
            ])
            self.cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.cnn_model.fit(train_gen, validation_data=val_gen, epochs=10)
        except Exception as e:
            print(f"Error training CNN model: {e}")

    def train_transfer_learning_model(self, image_dir=None, img_size=(64, 64), batch_size=32):
        try:
            if image_dir is None:
                image_dir = os.path.join(os.path.dirname(__file__), 'spectrograms')
            print(f"Loading images from: {image_dir}")
            datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
            train_gen = datagen.flow_from_directory(
                image_dir, target_size=img_size, batch_size=batch_size,
                class_mode='categorical', subset='training'
            )
            val_gen = datagen.flow_from_directory(
                image_dir, target_size=img_size, batch_size=batch_size,
                class_mode='categorical', subset='validation'
            )
            base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_size[0], img_size[1], 3))
            for layer in base_model.layers:
                layer.trainable = False
            self.transfer_model = Sequential([
                base_model,
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(train_gen.num_classes, activation='softmax')
            ])
            self.transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.transfer_model.fit(train_gen, validation_data=val_gen, epochs=10)
        except Exception as e:
            print(f"Error training transfer learning model: {e}")

def main():
    classifier = MusicGenreClassifier()

    print("="*50)
    print("TABULAR DATA APPROACH")
    print("="*50)
    data = classifier.load_tabular_data()
    if data is not None:
        X = data.drop(['label', 'filename'], axis=1)
        y = data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        classifier.train_tabular_model(X_train, X_test, y_train, y_test)

    print("="*50)
    print("CNN MODEL APPROACH")
    print("="*50)
    classifier.train_cnn_model()

    print("="*50)
    print("TRANSFER LEARNING MODEL APPROACH")
    print("="*50)
    classifier.train_transfer_learning_model()

if __name__ == "__main__":
    main()