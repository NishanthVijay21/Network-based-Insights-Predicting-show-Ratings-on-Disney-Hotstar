# Network-based-Insights-Predicting-show-Ratings-on-Disney-Hotstar

This repository contains a Jupyter Notebook dedicated to analyzing streaming platform data (specifically `disney_plus_titles.csv`). The project applies both Unsupervised Learning (K-Means Clustering) and Geometric Deep Learning (Graph Neural Networks) to extract insights and predict content ratings.

## 🚀 Features

### 1. Data Cleaning & Preprocessing
- Safely encodes categorical variables (e.g., converting "Movie" and "TV Show" into numerical formats).
- Extracts pure numerical data from text-heavy columns (e.g., converting `"90 min"` or `"1 Season"` into float values for the `duration` column).
- Handles missing values (`NaN`s) to ensure model stability.

### 2. K-Means Clustering
- Groups movies and TV shows into clusters based on their `type`, `release_year`, and `duration`.
- Visualizes the clusters and their calculated centroids using `matplotlib`.

### 3. Graph Neural Network (GNN) for Node Classification
- Constructs a mathematical graph where nodes represent individual movies/shows.
- Generates edges (connections) between movies that share the same genre (`listed_in`).
- Utilizes **PyTorch Geometric** to build a 2-layer Graph Convolutional Network (GCN).
- Trains the GCN to predict the `rating` of a show (e.g., PG-13, TV-MA) based on its features and its graph connections.

## 🛠️ Requirements

To run this notebook, you will need Python 3.8+ and the following libraries installed:

```bash
pip install pandas numpy matplotlib scikit-learn
pip install torch torchvision torchaudio  # Get the appropriate version for your OS/CUDA from pytorch.org
pip install torch_geometric
