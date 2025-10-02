import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import tensorflow
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class FAQModel:
    def __init__(self, data_file="data/intents.json"):
        with open(data_file, "r") as f:
            self.data = json.load(f)["intents"]
        
        self.model = None
        self.encoder = LabelEncoder()
        self.questions = []
        self.labels = []
    
    def train(self, epochs=300, batch_size=8):
        """Train a neural network on FAQ intents"""
        # Prepare data
        for intent in self.data:
            tag = intent["tag"]
            for q in intent.get("questions", []):
                self.questions.append(q.lower())
                self.labels.append(tag)
        
        # Encode labels
        y = self.encoder.fit_transform(self.labels)
        # Convert labels to one-hot
        y_onehot = np.zeros((len(y), len(set(y))))
        for i, val in enumerate(y):
            y_onehot[i, val] = 1
        
        # Vectorize questions using simple bag-of-words
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(self.questions).toarray()
        
        # Build Keras model
        self.model = Sequential()
        self.model.add(Dense(8, input_shape=(X.shape[1],), activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(y_onehot.shape[1], activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])
        
        # Train
        self.model.fit(X, y_onehot, epochs=epochs, batch_size=batch_size, verbose=1)
        return self
    
    def get_response(self, message):
        """Predict the intent and return answer"""
        if self.model is None:
            return "Model not trained yet."
        X_test = self.vectorizer.transform([message.lower()]).toarray()
        pred = self.model.predict(X_test)
        tag_idx = np.argmax(pred)
        tag_name = self.encoder.inverse_transform([tag_idx])[0]
        
        # Find the answer
        for intent in self.data:
            if intent["tag"] == tag_name:
                return intent.get("answer", "Sorry, I don't know.")
        return "Sorry, I don't know."
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    

    