# Iris Flower Clustering using K-Means

## Live Application
🌐 **Try the application here:**  
https://iris-flower-clustering.streamlit.app/

## Project Overview
This project uses the **K-Means Clustering algorithm**, an unsupervised machine learning technique, to group Iris flowers based on their physical measurements.

The model uses four features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The trained model and fitted StandardScaler are saved together and used by the Streamlit application to predict the cluster for new flower measurements.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Machine Learning Workflow
1. Load and explore the Iris dataset.
2. Select the four flower measurement features.
3. Remove duplicate records.
4. Standardize the features using StandardScaler.
5. Train a K-Means clustering model with 3 clusters.
6. Evaluate clustering performance using the Silhouette Score.
7. Save the trained model and scaler using Joblib.
8. Deploy the application using Streamlit Community Cloud.

## Application Features
- Simple and interactive user interface.
- Input fields for four Iris flower measurements.
- Predicts the cluster using the trained K-Means model.
- Uses the saved model and scaler for consistent predictions.
- Displays the prediction result instantly.

## Project Structure

```text
Iris-Flower-Clustering/
│
├── app.py
├── iris_kmeans_bundle.pkl
├── requirements.txt
└── README.md
```

## How to Run Locally

1. Clone this repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project folder:

```bash
cd Iris-Flower-Clustering
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Model Information
- **Algorithm:** K-Means Clustering
- **Learning Type:** Unsupervised Learning
- **Number of Clusters:** 3
- **Feature Scaling:** StandardScaler
- **Model Storage:** Joblib (.pkl)

## Project Objective
The main objective of this project is to understand and implement unsupervised machine learning using K-Means clustering and deploy the trained model as an interactive web application.

## Author
**Sarveswaran S.**

B.E. Artificial Intelligence and Machine Learning Student

---

⭐ If you find this project interesting, feel free to explore the live application!
