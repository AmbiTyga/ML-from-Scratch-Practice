import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LinearRegressor:
    def __init__(self, n_iters, lr=1e-3):
        self.n_iters = n_iters
        self.lr=lr
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, dim = X.shape

        self.weights = np.zeros(dim, dtype=np.float16)
        self.bias = 0

        for _ in range(self.n_iters):
            y_preds = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples)*np.dot(X.T, (y_preds-y))
            db = (1/n_samples)*np.sum((y_preds-y))
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self, X):
        y_preds = np.dot(X, self.weights) + self.bias
        return y_preds

if __name__ == "__main__":        
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1024)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2025
    )
    reg = LinearRegressor(1000, 0.01)
    reg.fit(X_train, y_train)

    preds = reg.predict(X_test)

    def mse(y_true, y_preds):
        return np.sum(np.square(y_true - y_preds))
    
    error = mse(y_test, preds)
    print(f"Error on test: {error}")