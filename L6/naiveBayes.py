import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
    
    def fit(self, X, y):
        n_samples, features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, features))
        self.var = np.zeros((n_classes, features))

        self.prob = np.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_ = X[y==c]

            self.mean[c, :] = X_.mean(axis = 0)
            self.var[c, :] = X_.var(axis=0)
            self.prob[c] = X_.shape[0]/n_samples
    
    def _pdf(self, X, idx):
        c1 = np.sqrt(2*np.pi*self.var[idx, :])
        c2 = np.exp(-(X-self.mean[idx, :])**2/(2*self.var[idx, :]))
        return np.log(c2/c1)
    
    def predict(self, X):
        probs = []
        for idx in self.classes:
            prob = np.sum(self._pdf(X, idx)) + np.log(self.prob[idx])
            probs.append(prob)
        return np.argmax(probs, 0)

if __name__ == "__main__":
    # imports
    from sklearn.model_selection import train_test_split as tts 
    from sklearn import datasets as ds 
    
    # create dataset
    np.random.seed(123)    
    X, y = ds.make_classification(n_samples=1000, n_features=10, n_classes=2)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2)
    
    # model
    lr = NaiveBayesClassifier()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    # evaluate
    acc = np.mean((y_test == y_pred))  
    print("Naive Bayes Test Accuracy:", acc)