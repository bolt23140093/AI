# coding: utf-8


from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# # Choosing a classification algorithm

# ...

# # First steps with scikit-learn

# Loading the Iris dataset from scikit-learn. Here, the third column represents the petal length, and the fourth column the petal width of the flower examples. The classes are already converted to integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.




iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

print('Class labels:', np.unique(y))


# Splitting data into 70% training and 30% test data:




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
"""test_size=0.3 資料30%用作test
在做切割前，會自動將資料打亂，以避免部分資料沒有被學習到
random_state=1 每次打亂資料結果會一致，它接受的值是 任何一個「整數」（int），理論上範圍不限
stratify=y	根據 y 的比例去切分資料，讓訓練集和測試集中，各類別的比例一樣。
（防止資料不平衡，比如 90% 是貓，10% 是狗的情況）
stratify=y 的意思是：依照 y 的類別比例進行分層抽樣。
這樣切完之後，訓練集 和 測試集 都會跟原來整體資料集中，各類別的比例差不多
"""


print('Labels count in y:', np.bincount(y))
print('Labels count in y_train:', np.bincount(y_train))
print('Labels count in y_test:', np.bincount(y_test))


# Standardizing the features:



#標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



# ## Training a perceptron via scikit-learn




ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
"""
項目	random_state    控制的是什麼？	要不要跟其他地方一樣？
train_test_split	 資料分割方式	不需要跟模型的 random_state 一樣
Perceptron	        初始權重的亂數	不需要跟資料切割的 random_state 一樣
"""

# **Note**
# 
# - You can replace `Perceptron(n_iter, ...)` by `Perceptron(max_iter, ...)` in scikit-learn >= 0.19. The `n_iter` parameter is used here deliberately, because some people still use scikit-learn 0.18.



y_pred = ppn.predict(X_test_std)
print('Misclassified examples: %d' % (y_test != y_pred).sum())
"""
第一行
用你剛剛訓練好的感知機模型 ppn，
拿測試集資料（標準化後的 X_test_std）來做預測，
把模型預測出來的類別結果存到 y_pred。

第二行
比對真實的標籤 y_test 和預測的結果 y_pred，
用 (y_test != y_pred) 建立一個布林陣列，例如：
[False, True, False, False, True, False]
True 代表這個樣本預測錯了。
.sum() 就是把 True 加總起來（True 當作1，False當作0）。
"""




print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
"""
用 accuracy_score 這個函數來計算模型的準確率。
y_test 是真實標籤，y_pred 是預測結果。
最後用 %.3f 的格式印出來（小數點後3位）。"""



print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))
"""
直接用感知機 ppn 的 .score() 方法來計算準確率。
ppn.score(X_test_std, y_test) 其實內部做的就是：
預測 (predict)
比對 (y_test vs 預測值)
計算準確率 (accuracy)
.score() 這個方法是包好的一鍵算準確率快捷方式！
"""




# To check recent matplotlib compatibility

"""
X: 特徵資料（兩個特徵，2D）
y: 標籤資料
classifier: 訓練好的分類器（比如這裡是 ppn）
test_idx: 要特別標示出來的測試資料索引
resolution: 網格解析度，決定畫得多細。"""
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #讓每個類別（class）有不同的顏色和形狀。
    #ListedColormap：設定一個小色盤，讓後面畫背景時用

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 #決定橫軸（特徵1）、縱軸（特徵2）的範圍，稍微留白（-1、+1）。
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution)) #建立一個網格（grid），每個小點是 (x1, x2) 的組合
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape) 
    #用模型預測每個網格點的類別。
    #ravel() 把矩陣攤平，reshape() 讓它又變回來。
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    #畫出網格的顏色背景，代表模型的分類結果。alpha=0.3：透明一點，可以看到資料點。

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    color=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black') #把每個類別的點用不同顏色和形狀畫出來。label=cl：給圖例（legend）用。

    # highlight test examples
    if test_idx:
        # plot all examples
        #如果有指定 test_idx，就把測試集資料圈起來。
        #c='none' 或 c=''：讓裡面是空心的。
        #edgecolor='black'：邊框是黑色。
        X_test, y_test = X[test_idx, :], y[test_idx]

        
        if LooseVersion(matplotlib.__version__) < LooseVersion('0.3.4'):
            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100, 
                        label='test set')
        else:
            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='none',
                        edgecolor='black',
                        alpha=1.0,
                        linewidth=1,
                        marker='o',
                        s=100, 
                        label='test set')        


# Training a perceptron model using the standardized training data:



X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_01.png', dpi=300)
plt.show()
"""感知器(Perceptron)缺點:如果不是線性會無法正確分類"""


# # Modeling class probabilities via logistic regression

# ...

# ### Logistic regression intuition and conditional probabilities





def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.arange(-7, 7, 0.1)
phi_z = sigmoid(z)

plt.plot(z, phi_z)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

# y axis ticks and gridline
plt.yticks([0.0, 0.5, 1.0])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
#plt.savefig('images/03_02.png', dpi=300)
plt.show()







# ### Learning the weights of the logistic cost function



def cost_1(z):
    return - np.log(sigmoid(z))


def cost_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10, 10, 0.1)
phi_z = sigmoid(z)

c1 = [cost_1(x) for x in z]
plt.plot(phi_z, c1, label='J(w) if y=1')

c0 = [cost_0(x) for x in z]
plt.plot(phi_z, c0, linestyle='--', label='J(w) if y=0')

plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('$\phi$(z)')
plt.ylabel('J(w)')
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('images/03_04.png', dpi=300)
plt.show()




class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.

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
      Logistic cost function value in each epoch.

    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
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
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # note that we compute the logistic `cost` now
            # instead of the sum of squared errors cost
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)





X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]

lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset,
         y_train_01_subset)

plot_decision_regions(X=X_train_01_subset, 
                      y=y_train_01_subset,
                      classifier=lrgd)

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
#plt.savefig('images/03_05.png', dpi=300)
plt.show()


# ### Training a logistic regression model with scikit-learn




lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr')
"""
這行程式碼創建了一個 LogisticRegression 的實例，並指定了 multi_class='ovr' 參數，這是處理多分類（例如三分類）的核心設定。
C=100.0：這是邏輯回歸模型的正則化參數。C 是正則化強度的倒數，越小的 C 表示更強的正則化。這裡的 C=100.0 是一個較小的值，表示相對較弱的正則化。
random_state=1：這個參數控制隨機數生成的種子，以保證每次運行的結果相同，特別是對於初始化的隨機權重等。
solver='lbfgs'：這是訓練模型所使用的優化算法。'lbfgs'（擬牛頓法）是一種高效的算法，常用於處理大規模數據和多分類問題。
multi_class='ovr'：這個參數設置了多分類的策略。'ovr'（一對其餘）是指將多分類問題拆解為多個二分類問題，每個分類器將某一個類別視為正類，其他所有類別視為負類。"""
lr.fit(X_train_std, y_train)
#將標準化後的訓練數據 X_train_std 和對應的目標標籤 y_train 傳遞給 LogisticRegression 模型，並開始進行模型的訓練。訓練過程會根據給定的參數，調整權重和參數來最佳化分類邏輯。
plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('images/03_06.png', dpi=300)
plt.show()




lr.predict_proba(X_test_std[:3, :])




lr.predict_proba(X_test_std[:3, :]).sum(axis=1)




lr.predict_proba(X_test_std[:3, :]).argmax(axis=1)




lr.predict(X_test_std[:3, :])




lr.predict(X_test_std[0, :].reshape(1, -1))



# ### Tackling overfitting via regularization

"""什麼是正則化（Regularization）？
正則化的目標是防止機器學習模型過擬合（overfitting）。
過擬合指的是模型在訓練資料上表現很好，但在新的、未見過的測試資料上表現很差，因為它「記太多細節」而不是「學到一般規則」。
正則化透過在「損失函數（loss function）」中加上懲罰項（penalty term），來限制模型的複雜度，讓模型的權重（w）不要變得太大。

Logistic Regression 中的正則化
在邏輯回歸中，我們通常會最小化下列目標函數：
Cost=Logistic Loss+λ×Penalty
Logistic Loss（邏輯損失）：就是模型預測錯誤的成本。
Penalty（懲罰項）：會懲罰太大的權重，常見的是 L2正則化（平方懲罰，即所有權重平方後相加）。
λ（lambda，懲罰強度）：決定懲罰的嚴格程度。
"""





weights, params = [], []
for c in np.arange(-5, 5):#C用不同範圍代入
    lr = LogisticRegression(C=10.**c, random_state=1,
                            solver='lbfgs',
                            multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
#plt.savefig('images/03_08.png', dpi=300)
plt.show()



# # Maximum margin classification with support vector machines





# ## Maximum margin intuition

# ...

# ## Dealing with the nonlinearly separable case using slack variables







#非線性喜好用SVM(先將2D投射到3D，分類完再將3D轉回2D)
#線性SVM不會做投射
svm = SVC(kernel='linear', C=1.0, random_state=1)
"""SVC：是 scikit-learn 裡的 Support Vector Classifier，用來做分類。
kernel='linear'：指定核函數（Kernel Function）為 線性，代表分隔邊界是直線（或高維的超平面）。
C=1.0：這是正則化參數，控制錯誤分類的容忍度。
C值小：容忍更多錯誤，決策邊界比較平滑。
C值大：不容忍錯誤，決策邊界更努力正確分類每個點。
random_state=1：隨機種子，確保結果可重現（每次執行結果一樣）。"""
svm.fit(X_train_std, y_train)
#X_train_std：是訓練用的特徵矩陣（已經標準化過，也就是均值0、標準差1）。
#y_train：是訓練用的目標標籤（真實分類，例如不同品種的花）。
plot_decision_regions(X_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))
"""X_combined_std：是訓練集和測試集合併後的特徵資料（同樣標準化過）。
            y_combined：是訓練集和測試集合併後的標籤。
            classifier=svm：用剛剛訓練好的 SVM 畫出邊界。
            test_idx=range(105, 150)：特別標示出 測試集的資料點，這些是第105到第149筆資料（通常是畫成不一樣的點形狀或顏色）。"""
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_11.png', dpi=300)
plt.show()


# ## Alternative implementations in scikit-learn




ppn = SGDClassifier(loss='perceptron')
lr = SGDClassifier(loss='log')
svm = SGDClassifier(loss='hinge')



# # Solving non-linear problems using a kernel SVM



#非線性
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.tight_layout()
#plt.savefig('images/03_12.png', dpi=300)
plt.show()







# ## Using the kernel trick to find separating hyperplanes in higher dimensional space


#非線性SVM
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
"""SVC：Support Vector Classifier。
kernel='rbf'：使用 RBF（高斯徑向基底）核函數 ➔ 可以處理非線性可分的資料
gamma=0.10：控制影響範圍：
小的gamma：模型更平滑，決策邊界較寬鬆。
大的gamma：模型擬合更細膩，決策邊界複雜。
C=10.0：正則化參數。
C大：希望每一個資料點都分類正確（少容錯），但可能過擬合。
C小：允許錯誤分類（簡化模型）。"""
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_14.png', dpi=300)
plt.show()





svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_15.png', dpi=300)
plt.show()




svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_16.png', dpi=300)
plt.show()



# # Decision tree learning










# ## Maximizing information gain - getting the most bang for the buck
#https://www.graphviz.org/download/
#在使用decision tree(決策樹)的時候會需要安裝的套件，否則會有圖形無法運作





def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))


def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))


def error(p):
    return 1 - np.max([p, 1 - p])

x = np.arange(0.0, 1.0, 0.01)

ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

fig = plt.figure()
ax = plt.subplot(111)
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err], 
                          ['Entropy', 'Entropy (scaled)', 
                           'Gini impurity', 'Misclassification error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=5, fancybox=True, shadow=False)

ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('impurity index')
#plt.savefig('images/03_19.png', dpi=300, bbox_inches='tight')
plt.show()



# ## Building a decision tree




tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)
"""DecisionTreeClassifier(...)：建立一個決策樹分類器。
criterion='gini'：使用 Gini impurity 作為分裂準則。
max_depth=4：設定最大深度為 4，防止過度擬合。
"""
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, 
                      classifier=tree_model,
                      test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()





tree.plot_tree(tree_model)
#plt.savefig('images/03_21_1.pdf')
plt.show()





#將tree產生PNG圖檔
dot_data = export_graphviz(tree_model,
                           filled=True, 
                           rounded=True,
                           class_names=['Setosa', 
                                        'Versicolor',
                                        'Virginica'],
                           feature_names=['petal length', 
                                          'petal width'],
                           out_file=None) 
graph = graph_from_dot_data(dot_data) 
graph.write_png('tree.png') 







# ## Combining weak to strong learners via random forests




forest = RandomForestClassifier(criterion='gini',
                                n_estimators=25, 
                                random_state=1,
                                n_jobs=2)
"""criterion='gini'：用 Gini impurity 作為每棵樹的分裂準則。
n_estimators=25：使用 25 棵決策樹。
random_state=1：設隨機種子確保重現性。
n_jobs=2：使用兩個 CPU 核心來加速訓練（可設為 -1 表示用盡所有核心）。"""
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, 
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_22.png', dpi=300)
plt.show()



# # K-nearest neighbors - a lazy learning algorithm








knn = KNeighborsClassifier(n_neighbors=5, 
                           p=2, 
                           metric='minkowski')
knn.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, 
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_24.png', dpi=300)
plt.show()



# # Summary

# ...

# ---
# 
# Readers may ignore the next cell.









