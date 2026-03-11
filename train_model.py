from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Create model
model = LogisticRegression(max_iter=200)

# Train model
model.fit(X, y)

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

print("Model training complete")
print("model.pkl file created")