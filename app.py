# ==========================================================
# Title: Flask Web App for Iris Flower Classification (Enhanced UI)
# ==========================================================

from flask import Flask, render_template, request   # Web framework + HTML rendering + form input
import pickle                                      # To load saved ML model
import numpy as np                                  # For numerical array handling

app = Flask(__name__)  # Initialize the Flask application

# ----------------------------------------------------------
# 1. Load the Pre-trained Machine Learning Model (.pkl file)
# ----------------------------------------------------------
# The model was already trained earlier and saved as 'iris_model.pkl'.
# During runtime, we simply load it instead of training again.
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# ----------------------------------------------------------
# 2. Static Information for Each Iris Species (Image + Description)
# ----------------------------------------------------------
# When prediction gives Setosa / Versicolor / Virginica, we display
# the corresponding image and description on the webpage.
iris_info = {
    "Setosa": {
        "image": "images/setosa.jpg",
        "desc": "Iris Setosa is a small, delicate flower with blue or violet petals and is native to northern regions."
    },
    "Versicolor": {
        "image": "images/versicolor.jpg",
        "desc": "Iris Versicolor has violet-blue petals and grows in wetlands. Also called the Blue Flag Iris."
    },
    "Virginica": {
        "image": "images/virginica.jpg",
        "desc": "Iris Virginica (Southern Blue Flag) is bigger and darker with bold purple-blue petals."
    }
}

# ----------------------------------------------------------
# 3. Routes (Pages of the Web App)
# ----------------------------------------------------------

@app.route("/")   # Home page
def home():
    return render_template("index.html")  # Just load the main HTML page initially


@app.route("/predict", methods=["POST"])  # Page triggered when user clicks "Predict"
def predict():
    try:
        # ---- Extract input values from the form (converted to float) ----
        sepal_length = float(request.form["sepal_length"])
        sepal_width  = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width  = float(request.form["petal_width"])

        # ---- Convert input to 2D array because model expects (n_samples, n_features) ----
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # ---- Perform prediction using ML model ----
        prediction = model.predict(features)[0]     # Output will be 0 / 1 / 2

        # ---- Convert numeric label to actual species name ----
        species = ["Setosa", "Versicolor", "Virginica"][prediction]

        # ---- Get the image + description for the predicted species ----
        info = iris_info[species]

        # ---- Send result back to webpage (UI) ----
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
        # If anything fails (missing input, invalid value, etc.)
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# ----------------------------------------------------------
# Run the Flask app (debug=True = show error logs live)
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
    