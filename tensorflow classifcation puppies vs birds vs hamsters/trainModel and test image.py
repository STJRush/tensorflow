import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define paths to your training and test (validation) image folders
train_dir = 'datasets/train/images'  # Replace with your actual path
test_dir = 'datasets/test/images'    # Replace with your actual path

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer with a neuron for each class
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=2,  # Depends on your dataset size
    epochs=15,
    validation_data=test_generator,
    validation_steps=2)  # Depends on your dataset size



from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(image_path):
    # Load the image file, targeting the size used during training
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale the image (as done in training)
    return img_array

def predict_image_class(model, image_path):
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Assuming you have a mapping of indices to class names
    class_names = list(train_generator.class_indices.keys())  # e.g., ['class1', 'class2', ...]
    predicted_class_name = class_names[predicted_class[0]]
    return predicted_class_name

# Example usage
test_image_path = 'datasets/test/my_test_images/test1.jpg'  # Replace with the path to your test image
predicted_class = predict_image_class(model, test_image_path)
print(f'This image is predicted to be a: {predicted_class}')
