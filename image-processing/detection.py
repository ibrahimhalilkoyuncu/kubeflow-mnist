import tensorflow as tf
print(tf.__version__)


mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


#normalizing
training_images  = training_images / 255.0
test_images = test_images / 255.0
index = 0

image = test_images[index]
label = test_labels[index]


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)


classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
index = 6

image = test_images[index]
label = test_labels[index]
import numpy as np
prediction = model.predict(image.reshape(1, 28, 28))
predicted_class = np.argmax(prediction)

print("Predicted:", class_names[predicted_class])
print("Actual:   ", class_names[label])

import matplotlib.pyplot as plt

plt.imshow(image, cmap='gray')
plt.title(
    f"Predicted: {class_names[predicted_class]}\n"
    f"Actual: {class_names[label]}"
)
plt.axis('off')
plt.show()