import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self, n_iters: int=1000, lr: float=1e-3):
        self.n_iters= n_iters
        self.lr = lr
        self.weights=None
        self.bias=None
    
    def fit(self, X: np.array, y:np.array):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float16)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_preds = np.dot(X, self.weights) + self.bias
            preds = sigmoid(linear_preds)
            dw = 1/n_samples * np.dot(X.T, preds - y)
            db = 1/n_samples * np.sum(preds - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
    def predict(self, X: np.array) -> np.array:
        linear_preds = np.dot(X, self.weights) + self.bias
        prob = sigmoid(linear_preds)
        preds = np.array([0 if i <=0.5 else 1 for i in prob])
        return preds
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=2025
    )
    clf = LogisticRegression(lr=0.01)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = (preds==y_test).sum()/y_test.shape[0]
    print(f"Accuracy of Logistic Regression on Breast Cancer Classificiation: {acc: 0.2f}")