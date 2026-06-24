1. Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(prediction)
2. Logistic Regression
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print(model.predict(X[:5]))
3. K-Nearest Neighbors (KNN)
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

print(model.predict(X[:5]))
4. Decision Tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

print(model.predict(X[:5]))
5. Random Forest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

print(model.predict(X[:5]))
6. Support Vector Machine (SVM)
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)

model = SVC(kernel="rbf")
model.fit(X, y)

print(model.predict(X[:5]))
7. Naive Bayes
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

X, y = load_iris(return_X_y=True)

model = GaussianNB()
model.fit(X, y)

print(model.predict(X[:5]))
8. Gradient Boosting
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)

model = GradientBoostingClassifier()
model.fit(X, y)

print(model.predict(X[:5]))
9. AdaBoost
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

X, y = load_iris(return_X_y=True)

model = AdaBoostClassifier()
model.fit(X, y)

print(model.predict(X[:5]))
10. Extra Trees
from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier

X, y = load_iris(return_X_y=True)

model = ExtraTreesClassifier()
model.fit(X, y)

print(model.predict(X[:5]))
11. Bagging Classifier
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

model = BaggingClassifier(estimator=DecisionTreeClassifier())
model.fit(X, y)

print(model.predict(X[:5]))
12. HistGradientBoosting
from sklearn.datasets import load_iris
from sklearn.ensemble import HistGradientBoostingClassifier

X, y = load_iris(return_X_y=True)

model = HistGradientBoostingClassifier()
model.fit(X, y)

print(model.predict(X[:5]))
13. Ridge Classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier

X, y = load_iris(return_X_y=True)

model = RidgeClassifier()
model.fit(X, y)

print(model.predict(X[:5]))
14. SGD Classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier

X, y = load_iris(return_X_y=True)

model = SGDClassifier(random_state=42)
model.fit(X, y)

print(model.predict(X[:5]))
15. Passive Aggressive Classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import PassiveAggressiveClassifier

X, y = load_iris(return_X_y=True)

model = PassiveAggressiveClassifier(random_state=42)
model.fit(X, y)

print(model.predict(X[:5]))
16. Perceptron
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

X, y = load_iris(return_X_y=True)

model = Perceptron()
model.fit(X, y)

print(model.predict(X[:5]))
17. MLP Neural Network
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier

X, y = load_iris(return_X_y=True)

model = MLPClassifier(hidden_layer_sizes=(100,))
model.fit(X, y)

print(model.predict(X[:5]))
18. Gaussian Process Classifier
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier

X, y = load_iris(return_X_y=True)

model = GaussianProcessClassifier()
model.fit(X, y)

print(model.predict(X[:5]))
19. Quadratic Discriminant Analysis (QDA)
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

X, y = load_iris(return_X_y=True)

model = QuadraticDiscriminantAnalysis()
model.fit(X, y)

print(model.predict(X[:5]))
20. Linear Discriminant Analysis (LDA)
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X, y = load_iris(return_X_y=True)

model = LinearDiscriminantAnalysis()
model.fit(X, y)

print(model.predict(X[:5]))
