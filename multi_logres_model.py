import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def load_images(base_path, num_images):
    images = []
    labels = []

    for i in range(num_images):
        if(os.path.exists(f"{base_path}/generated_image_{i}.png")):
            img = Image.open(f"{base_path}/generated_image_{i}.png")
            img_array = np.array(img).reshape(-1)
            label = base_path.split('/')[-1]
            match label:
                case "red":
                    label = 0
                case "yellow":
                    label = 1
                case "green":
                    label = 2
                case "blue":
                    label = 3
            images.append(img_array)
            labels.append(label)

    return np.array(images), np.array(labels)

#Softmax Activation used for multiclass classification
def softmax(z):
    exponential = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exponential / np.sum(exponential, axis=1, keepdims=True)

#Log Loss for multiclass classification
def log_loss(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) #We add a small value to prevent taking the log of 0, which could cause problems
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)#Used CCE equation, negation is used for optimization, compute the ean

class LogisticRegressionModel:
    def __init__(self, input_size, output_size, l1=0.01):
        self.weights = np.zeros((input_size, output_size))
        self.bias = np.zeros(output_size)
        self.loss_history = []
        self.l1 = l1

    def predict(self, X):
        predict_val = np.dot(X, self.weights) + self.bias
        return softmax(predict_val)

    #Preventing overfitting
    def calculate_regularization_term(self):
        return self.l1 * np.sum(np.abs(self.weights))
    
    def train(self, X, y, alpha=0.01, epochs=100):
        one_hot_labels = np.eye(len(np.unique(y)))[y] #one hot encoding trick I found

        #Training loop
        for epoch in range(epochs):
            y_pred = self.predict(X)
            loss = log_loss(one_hot_labels, y_pred)

            #taking the partial dervicative of L respect to weight
            delta_w = (np.dot(X.T,  y_pred - one_hot_labels) + self.l1 * np.sign(self.weights)) / len(y) #Multiplying the transpose of X with the y_pred - 
            delta_b = np.sum( y_pred - one_hot_labels, axis=0) / len(y)
            self.weights -= alpha * delta_w
            self.bias -= alpha * delta_b
            self.loss_history.append(loss)

        self.plot_loss()

    def plot_loss(self):
        plt.plot(range(len(self.loss_history)), self.loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.show()

def train_test_split(X, y, test_size=0.2, random_seed=None):
    if random_seed: #Helps with keeping it reproducible for testing
        np.random.seed(random_seed)

    indices = np.arange(len(y)) #creates an array of integers of 0
    np.random.shuffle(indices) #shuffles the array of indeces so that I can permute the elemetns of the array, helps for traiing and testing variability

    index = int((1 - test_size) * len(y)) #choosing the index to split the dataset into two seperate 
    X_train, X_test = X[indices[:index]], X[indices[index:]]
    y_train, y_test = y[indices[:index]], y[indices[index:]]

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    num_images = 5000

    #Loading images
    red_images, red_labels = load_images("dataset_2/red", num_images)
    yellow_images, yellow_labels = load_images("dataset_2/yellow", num_images)
    green_images, green_labels = load_images("dataset_2/green", num_images)
    blue_images, blue_labels = load_images("dataset_2/blue", num_images)

    X = np.concatenate([red_images, yellow_images, blue_images, green_images]) #Creating the 
    y = np.concatenate([red_labels, yellow_labels, blue_labels, green_labels])

    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0)) #Normalization 
    X = np.c_[X, np.ones(X.shape[0])]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

    INPUT_SIZE = X_train.shape[1]
    OUTPUT_SIZE = len(np.unique(y))  
    model = LogisticRegressionModel(INPUT_SIZE, OUTPUT_SIZE)
    model.train(X_train, y_train, alpha=0.01, epochs=10000)

    y_pred = model.predict(X_test)
    test_loss = log_loss(np.eye(OUTPUT_SIZE)[y_test], y_pred)
    print(f"Test Loss: {test_loss}")

    for i in range(len(y_pred)):
        predicted_class = np.argmax(y_pred[i])
        print(f"Sample {i}: Predicted Class: {predicted_class}, True Class: {y_test[i]}")