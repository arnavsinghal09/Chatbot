import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


# Load intents.json file
with open('intents.json') as file:
    data = json.load(file)

# Extract patterns and tags
patterns = []
tags = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)

# Tokenize patterns
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, maxlen=20)

# One-hot encode labels
labels = tf.keras.utils.to_categorical(labels)
# Define the CNN model
model = Sequential([
    Embedding(input_dim=2000, output_dim=32, input_length=20),
    Conv1D(filters=256, kernel_size=10, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(256, activation='relu'),
    Dense(len(data['intents']), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(padded_sequences, labels, epochs=30, batch_size=20, validation_split=0.2)



# Predict on training data
y_pred = model.predict(padded_sequences)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(labels, axis=1)

# Calculate metrics
report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_)
print(report)

accuracy = accuracy_score(y_true, y_pred_classes)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='weighted')

# Print metrics with six decimal places
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1 Score: {f1:.6f}")

def chatbot_response(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=20)
    pred = model.predict(padded_sequence)
    tag = label_encoder.inverse_transform([np.argmax(pred)])
    for intent in data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

# Example usage
print(chatbot_response("Hello"))
print(chatbot_response("Thank you"))
