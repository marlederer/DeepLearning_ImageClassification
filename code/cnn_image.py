import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import ssl
from sklearn import metrics
from keras.models import load_model

ssl._create_default_https_context = ssl._create_unverified_context

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Assuming x_train is a 4-dimensional tensor of shape (num_samples, 3, 32, 32)
# and y_train is a 1-dimensional tensor of shape (num_samples,)
# Convert numpy arrays to PyTorch tensors

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
loaded_model = None

try:
    # Load the model from the file
    loaded_model = load_model('richardParkman.h5')

    # Now, you can use the loaded_model for predictions, evaluation, etc.
    # For example:
    # loaded_model.predict(new_data)

except Exception as e:
    print(f"Error loading the model: {e}")

if loaded_model is None:
    history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
    model.save('richardParkman.h5')

"""
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
"""
if loaded_model is None:
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
else:
    test_loss, test_acc = loaded_model.evaluate(test_images, test_labels, verbose=2)

print(test_acc)

actual = test_labels
if loaded_model is None:
    predicted = model.predict(test_images)
else:
    predicted = loaded_model.predict(test_images)


y_pred_classes = np.argmax(predicted, axis=1)
print(y_pred_classes)

y_pred_classes = y_pred_classes.reshape(-1, 1)

print(actual.shape)
print(y_pred_classes.shape)

confusion_matrix = metrics.confusion_matrix(actual, y_pred_classes)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

print(cm_display)
cm_display.plot()
plt.show()
