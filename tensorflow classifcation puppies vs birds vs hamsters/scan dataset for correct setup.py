import os

train_dir = 'datasets/train/images'  # Replace with your actual path
test_dir = 'datasets/test/images'  # Replace with your actual path

print("Training directory contents:")
print(os.listdir(train_dir))

print("\nTest directory contents:")
print(os.listdir(test_dir))