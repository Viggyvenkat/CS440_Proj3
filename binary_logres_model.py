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
            label = 1 if "is_dangerous" in base_path else 0  # 1 for dangerous, 0 for safe
            images.append(img_array)
            labels.append(label)

    return np.array(images), np.array(labels)

#Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred, epsilon = 1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) #We add a small value to prevent taking the log of 0, which could cause problems
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean() #Used BCE equation, negation is used for optimization, compute the ean

class LogisticRegressionModel:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.loss_history = []

    def predict(self, X):
        predict_val = np.dot(X, self.weights) + self.bias #X dot product W + b
        return sigmoid(predict_val) #sigmoid(predict_val)

    def train(self, X, y, alpha=0.01, NUM_EPOCHS=100):
        for epoch in range(NUM_EPOCHS):
            y_pred = self.predict(X) #Forward pass
            loss = log_loss(y, y_pred) #Calculating log loss

            delta_w = np.dot(X.T, y_pred - y) / len(y) #calculating delta J(Theta)
            delta_b = np.sum(y_pred - y) / len(y) #Calculating the new bias value

            self.weights -= alpha * delta_w #updating the weights
            self.bias -= alpha * delta_b #updating the bias

            self.loss_history.append(loss) #for graphing
        
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
    NUM_IMAGES = 5000

    dangerous_images, dangerous_labels = load_images("dataset_1/is_dangerous", NUM_IMAGES)
    safe_images, safe_labels = load_images("dataset_1/safe", NUM_IMAGES)

    X = np.concatenate([dangerous_images, safe_images])
    y = np.concatenate([dangerous_labels, safe_labels])

    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0)) #Normalization
    X = np.c_[X, np.ones(X.shape[0])] #Adding a bias column 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

    input_size = X_train.shape[1]
    model = LogisticRegressionModel(input_size)
    model.train(X_train, y_train, alpha=0.01, NUM_EPOCHS=10000)

    y_pred = model.predict(X_test)
    test_loss = log_loss(y_test, y_pred)
    print(f"Test Loss: {test_loss}")

    for y in range(len(y_pred)):
        if y_pred[y] >= 0.5:
            print(f"Dangerous: predicted value:{ y_pred[y]},Real value:{y_test[y]}")
        else: 
            print(f"Safe: predicted value:{ y_pred[y]},Real value:{y_test[y]}")