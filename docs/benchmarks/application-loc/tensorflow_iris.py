import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) Load CSV
df = pd.read_csv("irisflowers.csv")
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.astype(np.float32)
y = pd.factorize(df["class"])[0].astype(np.int64)

# 2) Split + scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3) Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),
    tf.keras.layers.Dense(16, activation="tanh"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 4) Training
model.fit(X_train, y_train, epochs=100, verbose=0)

# 5) Confusion matrix
probs = model.predict(X_test, verbose=0)
preds = probs.argmax(axis=1)
cm = tf.math.confusion_matrix(y_test, preds, num_classes=3)

print("Confusion matrix:\n", cm.numpy())

# 6) Deployment
x = scaler.transform([[5.1, 3.5, 1.4, 0.2]]).astype(np.float32)
y = model(x, training=False).numpy().argmax(axis=1)

print("Predicted class:", int(y[0]))

# 7) Export
tf.saved_model.save(model, "iris_model")
