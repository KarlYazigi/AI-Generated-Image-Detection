
#Convolutional neural network (CNN) model for image classification trained on a NVIDIA RTX 2060. Code loads the training and #testing data, constructs the CNN model with specific layers and activations, performs training on the GPU, and finally saves #the trained model along with the accuracy results.

import tensorflow as tf
from tensorflow.keras import layers
from tqdm import tqdm
import pickle as pkl
import os
import gc

# Define the CNN model
def create_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(245, 255, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    return model

def train_model(model, train_data, test_data):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    results = []
    for epoch in range(10):
        total = 0
        
        for batch in train_data:
            train_X, train_Y = batch
            # Train model
            history = model.fit(train_X, train_Y, epochs=1, validation_data=test_data)
            gc.collect()
            tf.keras.backend.clear_session()
            
            # Save results
            results.append([history.history['val_accuracy'][0], history.history['accuracy'][0]])
    
    return results

# Load test set
with open('D:\\Image_Classifier\\batches\\test_batch\\test_batch.pickle', 'rb') as f:
    test_data = pkl.load(f)

# Load train set
train_data = []
batch_path = 'D:\\Image_Classifier\\batches\\train_batches\\'
for batch in os.listdir(batch_path):
    with open(batch_path + batch , 'rb') as f:
        train_data.append(pkl.load(f))

# Create and train the model
model = create_model()
results = train_model(model, train_data, test_data)

# Save the trained model
model.save('D:\Image_Classifier\\trained_model')

# Save the results
with open('D:\Image_Classifier\\accuracy.pickle', 'wb') as f:
    pkl.dump(results, f)

