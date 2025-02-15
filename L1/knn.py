import numpy as np
from collections import Counter
from sklearn import datasets 
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

class KNN:
    def __init__(self, k=3):
        self.k=k
        self.X=None
        self.y=None
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def _predict(self, x):
        distances = np.array([euclidean_distance(x, x_) for x_ in self.X])
        sorted_distances = np.argsort(distances)
        topK = sorted_distances[:self.k]
        labels = self.y[topK]
        label = Counter(labels.tolist()).most_common()
        return label[0][0]

    def predict(self, X):
        labels = []
        for x in X:
            labels.append(self._predict(x))
        return np.array(labels)

if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=2025, test_size=0.1
    )
    clf = KNN(5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.sum(y_pred==y_test)/y_test.shape[0]
    print(f"Accuracy of KNN: {acc}")