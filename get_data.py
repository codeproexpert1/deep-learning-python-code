import cv2
import numpy as np
import os

# Create directories for training and testing data
train_dir = 'synthetic_data/train'
test_dir = 'synthetic_data/test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Generate synthetic images for training
for i in range(1000):
    image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    label = np.random.randint(0, 2)
    cv2.imwrite(f'{train_dir}/{label}_{i}.jpg', image)

# Generate synthetic images for testing
for i in range(200):
    image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    label = np.random.randint(0, 2)
    cv2.imwrite(f'{test_dir}/{label}_{i}.jpg', image)
