import numpy as np
from typing import List
from tqdm.auto import tqdm, trange
import time
# MLP
class MLP:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        **kwargs
    ):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        sizes = [input_size]+hidden_sizes+[output_size]
        self.num_layers = len(hidden_sizes) - 1
        self.weights = []
        self.biases = []

        for i in range(1, self.num_layers+1):
            self.weights.append(
                np.random.randn(
                    sizes[i], sizes[i-1]
                )
            )
            self.biases.append(
                np.random.randn(sizes[i], 1)
            )

    # Forward Pass
    def forward(
        self,
        X
    ):
        self.activations = [X]
        self.z = []
        for i in range(self.num_layers):
            z = np.dot(
                self.weights[i], self.activations[-1]
            ) + self.biases[i]
            self.z.append(z)

            if i<self.num_layers-1:
                a = self.tanh(z)
            else:
                a = z
            self.activations.append(a)

        return self.activations[-1]
    
    # Backward Pass
    def backward(self, X, y):
        num_samples = X.shape[1]
        gradients = []
        dZ = self.activations[-1] - y # (num_layers+1)th
        for i in range(self.num_layers-1, -1, -1):
            dW = (1/num_samples) * np.dot(dZ, self.activations[i].T) # Difference * input
            db = (1/num_samples) * np.sum(dZ, axis=1, keepdims=True)
            gradients.append((dW, db))

            if i>0:
                dA = np.dot(self.weights[i].T, dZ)
                dZ = dA * self.gradient_tanh(self.z[i-1]) 
        return gradients[::-1]
    
    def update_parameters(self, gradients, lr):
        for i in range(self.num_layers):
            self.weights[i] -= lr*gradients[i][0]
            self.biases[i] -= lr*gradients[i][1]
    
    def tanh(self, Z):
        return np.tanh(Z)
    
    def gradient_tanh(self, Z):
        return 1- np.tanh(Z)**2
    

if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(
        n_samples=1000,
        n_features=1,
        noise = 0.3,
        random_state=2025
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=2025
    )

    X_train_mean = np.mean(X_train)
    X_train_std = np.std(X_test)

    X_train = (X_train - X_train_mean)/X_train_std
    X_test = (X_test - X_train_mean)/X_train_std
    
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    input_size = X_train.shape[1]
    hidden_sizes = [10, 10]
    output_size = y_train.shape[1]
    mlp = MLP(
        input_size,
        hidden_sizes,
        output_size
    )

    num_epochs = 1000
    learning_rate = 1e-2
    pbar = trange(num_epochs)
    for epoch in pbar:
        outputs = mlp.forward(X_train.T)

        gradients = mlp.backward(X_train.T, y_train.T)

        mlp.update_parameters(gradients, learning_rate)

        loss = np.mean(
            (outputs - y_train.T)**2
        )
        time.sleep(0.01)
        # if (epoch+1)%100 == 0:
        pbar.set_postfix(Loss=loss)
        pbar.set_description(f"Epoch {epoch}")
    
    test_outputs = mlp.forward(X_test.T)
    test_loss = np.mean((test_outputs - y_test.T)**2)
    print(f"Test Loss: {test_loss}")