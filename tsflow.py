import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Simple spam vs normal email dataset

# Features:
# [contains_link, contains_money_words]
x = tf.constant([
    [1, 1],  # "Win cash now" -> spam
    [0, 1],  # "Claim your offer" -> spam
    [1, 0],  # "Click the link" -> spam
    [0, 0],  # "Meeting tomorrow" -> normal
    [0, 0],  # "Lunch at 12?" -> normal
    [1, 0],  # "Visit my page" -> spam
    [0, 1],  # "Earn points today" -> spam
    [0, 0], #Can we talk
])
y = tf.constant([
    [1],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
    [0],
])

model = keras.Sequential([
    layers.Dense(5, activation="relu", input_shape=(2,)),
    layers.Dense(1, activation="sigmoid")
    ])

optimizer = keras.optimizers.Adam(learning_rate=0.05)  # or even 0.001
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])


model.fit(x, y, epochs=20, batch_size=1, verbose=2)

print(model.predict(tf.constant([[0, 0]])))
print(model.predict(tf.constant([[1, 0]])))
print(model.predict(tf.constant([[1, 1]])))
print(model.predict(tf.constant([[0, 1]])))
