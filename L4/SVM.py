import numpy as np

## Task:
# w.x_i - b>= 1 if y_i = 1
# w.x_i - b<= -1 if y_i = -1

# Loss function:
## Hinge Loss
## l = max(0, 1-y_i(w.x_i - b))
## l = 0 if y.f(x) >= 1
## l = 1- y.f(x) otherwise

## Cost Function:
# J = lambda.||w||^2 + L/n
# J_i = lambda.||w||^2 if y_i.f(x) >=1
# J_i = lambda.||w||^2 + 1-y_i(w.x_i - b)

## Derivatives:
## if y_i.f(x) >=1:
# dw = 2.lambda.w
# db = 0
## else:
# dw_k = w.lambda.w_k - y_i.x_ik
# db = y_i

## Update rule:
## if y_i.f(x) >=1:
# w = w - lr.dw = w - lr.2.lambda.w
# b = b - lr.b = b
## else:
# w = w - lr.dw = w - lr.(2.lambda.w - y_i.x_i)
# b = b - lr.db = b - lr.y_i

class SVM:
    def __init__(
            self, lr: float = 1e-3, 
            lambda_para:float = 1e-2,
            n_iters: int = 1000
    ):
        self.lr = lr
        self.lambda_para = lambda_para
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y<0, -1, 1)

        #init weights
        self.w = np.random.randn(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >=1
                if condition:
                    self.w -= self.lr * self.lambda_para * self.w * 2
                else:
                    self.w -= self.lr * (self.lambda_para * self.w * 2 - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]



    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
    
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(
        n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )

    y = np.where(y==0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2025
    )

    clf = SVM()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    def accuracy(y_true, y_pred):
        acc = np.sum(y_true==y_pred) / len(y_true)
        return acc
    
    print("SVM Classification accuracy: ", accuracy(y_test, preds))

    def visualize():
        def get_hyperplace(x, w, b, offset):
            return (-w[0]*x + b + offset)/ w[1]
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c= y)
        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplace(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplace(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplace(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplace(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplace(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplace(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])
        plt.show()

    visualize()