# ==========================================================
# Title: Flask Web App for Iris Flower Classification (Enhanced UI)
# Author: Lohith's Learning Series
# ==========================================================

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ----------------------------------------------------------
# 1. Load the Pre-trained Model
# ----------------------------------------------------------
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------------------------------------
# 2. Define Info for Each Iris Species
# ----------------------------------------------------------
iris_info = {
    "Setosa": {
        "image": "images/setosa.jpg",
        "desc": "Iris Setosa is a small, delicate flower with blue or violet petals and is native to northern regions."
    },
    "Versicolor": {
        "image": "images/versicolor.jpg",
        "desc": "Iris Versicolor has rich violet-blue petals and usually grows in wetlands. It's also called the Blue Flag Iris."
    },
    "Virginica": {
        "image": "images/virginica.jpg",
        "desc": "Iris Virginica, known as the Southern Blue Flag, is larger and darker with elegant purple-blue petals."
    }
}

# ----------------------------------------------------------
# 3. Routes
# ----------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        # Model prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)[0]
        species = ["Setosa", "Versicolor", "Virginica"][prediction]

        # Get info for that species
        info = iris_info[species]

        return render_template(
            "index.html",
            prediction_text=f"The predicted Iris species is: {species}",
            image_path=info["image"],
            desc=info["desc"],
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width
        )
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
