from fastapi import FastAPI
import pickle
import numpy as np

# create FastAPI app
app = FastAPI()

# load trained model
model = pickle.load(open("model.pkl", "rb"))

# home route
@app.get("/")
def home():
    return {"message": "ML API is running"}

# prediction route
@app.post("/predict")
def predict(sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float):

    data = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

    prediction = model.predict(data)

    return {"prediction": int(prediction[0])}