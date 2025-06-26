# coding: utf-8


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# # Artificial neurons - a brief glimpse into the early history of machine learning





# ## The formal definition of an artificial neuron





# ## The perceptron learning rule



# # Implementing a perceptron learning algorithm in Python

# ## An object-oriented perceptron API





class Perceptron(object):#機器學習模型
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y): #X=feature, Y=Label
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)#確保每次實驗「隨機初始化」是一樣的
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])#初始化權重向量（小小的隨機數），有 1 + 特徵數量 個元素。（因為有一個是 bias 截距項）
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            """對每一個樣本 (xi, target)：
                self.predict(xi)：用目前的權重預測 xi 的標籤。
                (target - self.predict(xi))：看預測錯多少。
                update = 學習率 * (目標值 - 預測值)：計算「調整量」。
                更新權重：self.w_[1:] += update * xi
                更新偏置（bias）：self.w_[0] += update
                如果有更新（update ≠ 0），就記一次錯誤。"""
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)




v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
"""
v1.dot(v2)：計算 v1 和 v2 的內積。
np.linalg.norm(v1)：計算 v1 的長度（L2 norm）。
np.linalg.norm(v2)：計算 v2 的長度。
所以 v1.dot(v2) / (||v1|| * ||v2||) 是什麼？👉 它就是cos(θ)，也就是兩向量之間夾角的餘弦值。
np.arccos(...)：再用反餘弦（arccos）把 cos(θ) 轉回真正的角度（弧度單位）。
總結來說：這一行就是算出「v1 和 v2 之間的夾角 θ（弧度）」。
"""


# ## Training a perceptron model on the Iris dataset

# ...

# ### Reading-in the Iris data





s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', s)

df = pd.read_csv(s,
                 header=None,
                 encoding='utf-8')

df.tail()


# 
# ### Note:
# 
# 
# You can find a copy of the Iris dataset (and all other datasets used in this book) in the code bundle of this book, which you can use if you are working offline or the UCI server at https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data is temporarily unavailable. For instance, to load the Iris dataset from a local directory, you can replace the line 
# 
#     df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#         'machine-learning-databases/iris/iris.data', header=None)
#  
# by
#  
#     df = pd.read_csv('your/local/path/to/iris.data', header=None)
# 



df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.tail()




# ### Plotting the Iris data




# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
"""
np.where(條件, 如果是True的結果, 如果是False的結果)
y == 'Iris-setosa'：檢查每個 y 裡的元素是不是 'Iris-setosa'
如果是 'Iris-setosa'，就把它變成 -1
如果不是 'Iris-setosa'，就變成 1
"""

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

# plt.savefig('images/02_06.png', dpi=300)
plt.show()



# ### Training the perceptron model



ppn = Perceptron(eta=0.1, n_iter=10)
#建立一個 Perceptron 物件，設定學習率(每次調整的幅度) η = 0.1，訓練次數 10 回合
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
"""
把 每一個 epoch 犯錯的次數 (ppn.errors_) 畫出來！
range(1, len(ppn.errors_) + 1)：代表第 1 回合、第 2 回合、...一直到第 10 回合。
marker='o'：每個點上畫一個小圈圈，方便看清楚每回合的錯誤數量"""
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)
plt.show()



# ### A function for plotting decision regions




#畫決策邊界
def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])#ListedColormap：建立一個「顏色對應表」，不同類別用不同顏色

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    #(定義畫圖範圍)找出第1維、第2維特徵的最小最大值，加減一點 margin，讓圖不會太擠
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #np.meshgrid：在整個範圍內鋪一層細網格。
    #resolution 是你每隔多少距離取一個點（數值越小，網格越細、圖越平滑）。
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    """把網格展平 (ravel) 後餵進 classifier 預測。
        Z 就是每個網格點被分類到哪一類。
        然後 reshape 回原本的網格形狀，方便畫等高線。"""
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)#contourf：畫出各區域填色（不同類別不同顏色），透明度 0.3，疊在資料點下面。
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())#讓座標軸範圍剛好包住所有網格。

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
        """np.unique(y)：找出所有類別。
            for idx, cl in enumerate(...)：對每個類別一個一個畫。
            X[y == cl, 0] & X[y == cl, 1]：只取該類別的點。
            alpha=0.8：點點的透明度。
            c=colors[idx] & marker=markers[idx]：每個類別配一種顏色和形狀。"""

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


# plt.savefig('images/02_08.png', dpi=300)
plt.show()



# # Adaptive linear neurons and the convergence of learning

# ...

# ## Minimizing cost functions with gradient descent





# ## Implementing an adaptive linear neuron in Python



class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])#初始化權重：self.w_（小小亂數）
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,  
            # in the case of logistic regression (as we will see later), 
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
            """計算 net input（加權總和）
                計算 output（這邊是線性函數，所以 output == net_input）
                計算 error：errors = (y - output)
                更新權重：
                self.w_[1:] += self.eta * X.T.dot(errors)
                self.w_[0] += self.eta * errors.sum()
                記錄 cost（SSE 平方誤差和）cost越小越準"""
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]#計算 加權總和：𝑧=𝑤0+𝑤1𝑥1+𝑤2𝑥2+…z=w0​ +w1​ x1 +w2 x2 +…

    def activation(self, X):
        """Compute linear activation"""
        return X#這裡是 線性函數（直接傳回來）， （為什麼要留這個？ → 因為以後如果想改成別的，例如 Logistic 函數，只要改這裡就好了。）

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
        #根據 output：如果 ≥ 0，就預測成 1 否則預測成 -1



fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.savefig('images/02_11.png', dpi=300)
plt.show()

"""
簡單來說，學習率太小 → 學得太慢，
學習率剛好 → 又快又穩定，
學習率太大 → 可能會爆炸震盪。(如圖中eta=0.01)"""






# ## Improving gradient descent through feature scaling







# standardize features 標準化
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()



#標準化之後比較不需要擔心eta值過大
ada_gd = AdalineGD(n_iter=15, eta=0.01)
ada_gd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/02_14_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

plt.tight_layout()
# plt.savefig('images/02_14_2.png', dpi=300)
plt.show()



# ## Large scale machine learning and stochastic gradient descent



class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    shuffle : bool (default: True)
      Shuffles training data every epoch if True to prevent cycles.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value averaged over all
      training examples in each epoch.

        
    """
    """是 Adaline 的一種改良版，
        改成一次只用一筆資料更新權重（而不是像 AdalineGD 那樣，每個 epoch 累積完一整批資料再更新）。
        支援：
        每 epoch 洗牌 (shuffle=True)，避免資料順序影響學習。
        partial_fit() → 局部更新，可以隨時丟新資料來微調。
        適合資料量大"""
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)




ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('images/02_15_1.png', dpi=300)
plt.show()

plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

plt.tight_layout()
# plt.savefig('images/02_15_2.png', dpi=300)
plt.show()




ada_sgd.partial_fit(X_std[0, :], y[0])



# # Summary

# ...

# --- 
# 
# Readers may ignore the following cell




