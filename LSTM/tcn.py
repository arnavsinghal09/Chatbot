import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load intents.json
with open('intents.json') as file:
    data = json.load(file)

# Extract sentences and labels
sentences = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
label_sequences = label_encoder.fit_transform(labels)

# Encode sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
max_len = max([len(x) for x in sequences])

# Padding sequences
X = pad_sequences(sequences, maxlen=max_len)
y = pd.get_dummies(label_sequences).values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_tcn_model(vocab_size, max_len, num_classes):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_len))

    # Adding dilated causal convolutional layers
    for dilation_rate in (1, 2, 4, 8):
        model.add(Conv1D(64, kernel_size=3, padding='causal', dilation_rate=dilation_rate, activation='relu'))
        model.add(Dropout(0.2))

    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = build_tcn_model(vocab_size, max_len, y.shape[1])
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=55, batch_size=16, validation_split=0.2)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average='weighted')
recall = recall_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
