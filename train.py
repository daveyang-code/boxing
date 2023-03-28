import tensorflow as tf
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import os
import numpy as np

actions = np.array(["jab", "cross", "hook", "uppercut"])
DATA_PATH = os.path.join("/home/david/Documents/Projects/boxing/keypoints")

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []

for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(16):
            res = np.load(
                os.path.join(
                    DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)
                )
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


model = tf.keras.models.Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(16, 132)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(actions.shape[0], activation="softmax"))

model.compile(
    optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)

model.fit(X_train, y_train, epochs=2000)

model.summary()

model.save("boxing.h5")