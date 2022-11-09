import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

def euclidean_distance(a, b):
    # эвклиидово расстояние
    return np.linalg.norm(a - b, ord=2)

def normalize_data(X):
    # способ масштабирования через выборочную среднюю
    mean, std = X.mean(axis=0), X.std(axis=0)
    return (X - mean) / std, (mean, std)
  
def argsort(a):
    # сортировка данных, которая возвращет индексы массивов в отсортированном порядке
    return np.array(a).argsort()

def accuracy(y_true, y_pred):
    # эффективность обучения
    return np.sum(y_true == y_pred) / len(y_true)

class kNearestNeighbor():
    def __init__(self, k=3, dist_metric='euclidean', norm=True):
        self.k = k
        self.isFit = False
        self.norm = norm
        self._set_dist_func(dist_metric)

    def _set_dist_func(self, dist_metric):
        implemented_metrics = {'euclidean': euclidean_distance, }

        self.dist_func = implemented_metrics[dist_metric]

    def normalize_new(self, X_new):
        return (X_new - self.trn_mean) / self.trn_std
    
    def fit(self, X_train, y_train, v=False):
        if self.norm:
            X_train, (trn_mean, trn_std) = normalize_data(X_train)
            self.trn_mean = trn_mean
            self.trn_std = trn_std
        self.X_train = X_train
        self.y_train = y_train

        y_train_pred, y_train_pred_proba = [], []
        for i, x_i in enumerate(X_train):
            distances = []
            for j, x_j in enumerate(X_train):
                if i == j:
                    dist_ij = 0
                else:
                    dist_ij = self.dist_func(x_i, x_j)

                distances.append(dist_ij)

            pred_i = self.estimate_point(distances, y_train)
            y_train_pred_i, y_train_pred_proba_i = pred_i

            y_train_pred.append(y_train_pred_i)
            y_train_pred_proba.append(y_train_pred_proba_i)

    def estimate_point(self, distances, y):
        sort_idx = argsort(distances)
        y_closest = y[sort_idx][:self.k]
        
        most_common = Counter(y_closest).most_common(1)[0]
        y_pred_i = most_common[0]
        y_pred_proba_i = most_common[1] / len(y_closest)
        return y_pred_i, y_pred_proba_i

    def predict(self, X_new):
        if self.norm:
            X_new = self.normalize_new(X_new)

        y_new_pred, y_new_pred_proba = [], []

        for i, x_i in enumerate(X_new):
            distances = []
            for j, x_j in enumerate(self.X_train):
                dist_ij = self.dist_func(x_i, x_j)
                distances.append(dist_ij)

            pred_i = self.estimate_point(distances, self.y_train)
            y_pred_i, y_pred_proba_i = pred_i
            y_new_pred.append(y_pred_i)
            y_new_pred_proba.append(y_pred_i)

        return y_new_pred

iris = datasets.load_iris()

X = iris.data  
y = iris.target
k = 8

X_trn_, X_test_, y_trn, y_test = train_test_split(X, 
                                                 y, 
                                                 test_size=0.333, 
                                                 random_state=0,
                                                 stratify=y)

feature_idxs = [1, 3] 
classes = list(set(y))
legend = ['Setosa', 'Versicolour', 'Virginica']

feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
x_label, y_label = feature_names[feature_idxs[0]], feature_names[feature_idxs[1]] 

h = .05
pad = 0.5
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
colours = ['red', 'green', 'blue']
n_features = X.shape[1]

fig, axs = plt.subplots(4, 4, figsize=(20, 20), sharex='col', sharey='row',)

for row in range(n_features):
    for col in range(n_features):
        print(row, col)

        feature_idxs = [col, row, ]
        xlbl, ylbl = feature_names[feature_idxs[0]], feature_names[feature_idxs[1]] 
        X_trn, X_test = X_trn_[:, feature_idxs], X_test_[:, feature_idxs]

        print("Features: {}, {}".format(feature_names[feature_idxs[0]], feature_names[feature_idxs[1]]))
        
        knn = kNearestNeighbor(k=k)
        knn.fit(X_trn, y_trn, v=False)

        y_trn_pred = knn.predict(X_trn)
        trn_acc = accuracy(y_trn_pred, y_trn)

        y_test_pred = knn.predict(X_test)
        test_acc = accuracy(y_test_pred, y_test)

        print('\t train accuracy: {}'.format(trn_acc))
        print('\t test accuracy: {}'.format(test_acc))

        x_min, x_max = X[:, feature_idxs[0]].min() - pad, X[:, feature_idxs[0]].max() + pad
        y_min, y_max = X[:, feature_idxs[1]].min() - pad, X[:, feature_idxs[1]].max() + pad
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z).reshape(xx.shape)

        axs[row, col].pcolormesh(xx, yy, Z, cmap=cmap_light)

        for i in classes:
            idx = np.where(y_trn == classes[i])
            axs[row, col].scatter(X_trn[idx, 0], 
                        X_trn[idx, 1], 
                        c=colours[i], 
                        label=legend[i],
                        marker='o', s=20)


        if row==n_features-1:
            axs[row, col].set_xlabel(xlbl, fontsize=16)
            
        if col==0:
            axs[row, col].set_ylabel(ylbl, fontsize=16)
        axs[row, col].set_title("trn acc {}, test acc {}".format(trn_acc, test_acc))

fig.suptitle('Iris: kNN (k={})'.format(k), fontsize=26)
plt.show()