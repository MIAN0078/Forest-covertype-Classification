# 🌲 Forest Cover Type Classification

A Machine Learning project to predict the **forest cover type** (the predominant kind of tree cover) from cartographic variables.
This project uses classification models to identify the correct forest cover type based on features like elevation, aspect, slope, soil type, and wilderness area.
--
##  Project Overview

The goal of this project is to classify forest cover types into **7 categories** using cartographic data.
This is a supervised learning problem where the target variable is the forest cover type, and the features include both continuous and categorical variables.

##  Dataset

* **Source:** [Kaggle - Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction)
* **Size:** ~500,000 samples
* **Features:**
  * Elevation, Aspect, Slope
  * Horizontal & Vertical Distances
  * Hillshade indices
  * Soil Types (40 binary features)
  * Wilderness Area (4 binary features)
* **Target:** 7 Cover Types (integer labels 1–7)
  

##  Installation

Clone the repository and install dependencies:

    bash
git clone https://github.com/your-username/forest-cover-classification.git
cd forest-cover-classification
pip install -r requirements.txt


##  Usage

1. Place the dataset (`train.csv`, `test.csv`) in the `data/` directory.
2. Run preprocessing and training:

```bash
python train.py
```
3. Evaluate the model:

```bash
python evaluate.py

##  Models Used

* Logistic Regression
* Random Forest Classifier
* XGBoost
* Neural Networks (optional, for deep learning experiments)

Feature scaling, PCA, and hyperparameter tuning (GridSearchCV) were applied to improve results.

** Results**

* Achieved **~80-85% accuracy** using Random Forest and XGBoost.
* Neural Networks performed comparably with tuning.

Example Confusion Matrix:

![Confusion Matrix](assets/confusion_matrix.png)

Accuracy & Loss curves (for NN):

![Accuracy Curve](assets/accuracy_curve.png)

##  Future Work

* Try advanced deep learning models (CNN/TabNet for tabular data).
* Feature engineering for soil/wilderness data.
* Hyperparameter optimization with Optuna.
* Deployment using **Streamlit** or **Flask**.

##  Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

##  License

This project is licensed under the **MIT License**.

## Acknowledgments
* [Kaggle - Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction)
* scikit-learn, XGBoost, TensorFlow/PyTorch
* Open-source community
