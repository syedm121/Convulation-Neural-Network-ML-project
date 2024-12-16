
# CNN for Image Classification

This repository contains the implementation of a Convolutional Neural Network (CNN) for image classification using TensorFlow and Keras. The CNN is designed to classify images into categories (e.g., Cats and Dogs) and is demonstrated step-by-step.

---

## Steps to Use

### Step 1: Import Libraries

```python
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image
import os
```

### Step 2: Data Preparation

Prepare your dataset in the following structure:

```
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

Then, use the following code to load and preprocess the data:

```python
# Initialize Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Dynamically construct dataset paths
current_dir = os.getcwd()
train_path = os.path.join(current_dir, 'dataset', 'training_set')
test_path = os.path.join(current_dir, 'dataset', 'test_set')

# Load Training Data
training_set = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Load Test Data
test_set = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
```

### Step 3: Build the CNN

```python
# Initialize the CNN
cnn = tf.keras.models.Sequential()

# Add Convolutional and Pooling Layers
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the Layer
cnn.add(Flatten())

# Add Fully Connected Layers
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))

# Compile the Model
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Step 4: Train the Model

```python
# Train the model
cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=25
)
```

### Step 5: Evaluate and Test

```python
# Load and preprocess a single image
sample_image_path = os.path.join(test_path, 'sample_image.jpg')
sample_image = image.load_img(sample_image_path, target_size=(64, 64))
sample_image_array = image.img_to_array(sample_image)
sample_image_array = np.expand_dims(sample_image_array, axis=0)

# Make a prediction
result = cnn.predict(sample_image_array)
training_set.class_indices  # Display class indices

# Print the result
print("Prediction:", "Cat" if result[0][0] == 1 else "Dog")
```

---

## Medium Article

For a detailed explanation of this implementation, check out the Medium article:

**[Convolutional Neural Networks (CNNs) for Image Classification](https://medium.com/@syedm.upwork/convolutional-neural-networks-cnns-for-image-classification-4391d05e9ba0)**

---

Happy coding!
