📌 Project Overview

This project is a prototype AI system that simulates monitoring Instagram content and automatically classifies posts as approved or not approved based on their captions.

⚡ Features

Works fully in Google Colab.

Simulates Instagram captions with a dataset of approved and not approved posts.

Trains a Logistic Regression classifier using TF-IDF features.

Classifies new captions as Approved or Not Approved.

Visualizes the distribution of approved vs rejected posts.

🛠 Tools and Technologies

Python 3

Google Colab

Libraries:

pandas – for data handling

scikit-learn – for ML (Logistic Regression, TF-IDF)

matplotlib – for visualization

📝 How It Works

Dataset: A simulated set of Instagram captions is created with labels (1 = approved, 0 = not approved).

Text Processing: Captions are converted into numerical features using TF-IDF.

Training: Logistic Regression model is trained to predict approval.

Prediction: New captions can be tested for approval or rejection.

Visualization: Bar chart shows the number of approved vs not approved posts in the dataset.
