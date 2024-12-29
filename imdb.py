import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load IMDb data
data = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# Print the first training example
print(train_data[0])

# Preprocess word indices
word_index = data.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Decode review function
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

print(decode_review(test_data[0]), len(test_data[1]))

# Model creation
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))  # Embedding layer for words
model.add(keras.layers.GlobalAveragePooling1D())  # Global pooling to reduce the dimensions
model.add(keras.layers.Dense(16, activation="relu"))  # Fully connected layer
model.add(keras.layers.Dense(1, activation="sigmoid"))  # Output layer for binary classification
model.summary()

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Prepare training and validation sets
x_val = train_data[:10000]
x_train = train_data[10000:]
y_val = train_labels[:10000]
y_train = train_labels[10000:]

# Pad sequences to ensure consistent input length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=word_index["<PAD>"], padding="post", maxlen=256)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, value=word_index["<PAD>"], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)

# Train the model
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# Evaluate model on test data
results = model.evaluate(test_data, test_labels)
print("Test results:", results)

# Predict on a single review
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))

# Review encoding function for custom reviews
def review_encode(s):
    encoded = [1]  # Start token
    for word in s.split():
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)  # Unknown word token
    return encoded

# Model prediction on custom reviews
model = keras.models.load_model("model.h5")

with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        # Clean and preprocess the review text
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "")
        encoded_line = review_encode(nline)
        
        # Pad the sequence before prediction
        encoded_line = keras.preprocessing.sequence.pad_sequences([encoded_line], value=word_index["<PAD>"], padding="post", maxlen=256)
        
        # Get the prediction
        prediction = model.predict(encoded_line)
        
        # Print the results
        print(f"Review: {line}")
        print(f"Encoded: {encoded_line}")
        print(f"Prediction: {prediction[0]}")
