import tensorflow as tf
from tensorflow import keras

# Simple spam vs normal email dataset

# Features:
# [contains_link, contains_money_words]
x = tf.constant(
    [
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1],
        [0, 0], 
    ]
)
y = tf.constant(
    [
        [1],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
    ]
)

model = keras.Sequential(
    [
        keras.layers.Dense(5, activation="relu", input_shape=(2,)),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=0.05)  # or even 0.001
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])


model.fit(x, y, epochs=20, batch_size=1, verbose=2)

print(model.predict(tf.constant([[0, 0]])))
print(model.predict(tf.constant([[1, 0]])))
print(model.predict(tf.constant([[1, 1]])))
print(model.predict(tf.constant([[0, 1]])))
