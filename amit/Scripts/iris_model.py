# train_model.py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pickle

iris = load_iris()
X, y = iris.data, iris.target

model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
