# Fight Song Analysis
Machine learning models to cluster and analyze fight songs, using K-Means and a Decision Tree Classifier for pattern prediction

## Overview
This Python script leverages machine learning algorithms to analyze and cluster data based on numerical and binary features extracted from a dataset, presumably related to fight songs, using libraries such as Pandas, NumPy, Scikit-learn, and Matplotlib. Here's a breakdown of the script's structure and goals:

### 1. Data Import and Preprocessing:
The script imports necessary libraries and defines two lists of feature types: `numerical_features` and `binary_features`. It reads data from a CSV file ("fight-songs.csv") and preprocesses it by replacing categorical binary values (Yes, No, Unknown) with numerical codes (1 for Yes and 0 for No/Unknown). Both feature types are then scaled using `StandardScaler` to ensure that all features contribute equally to the model without bias due to their scale.

### 2. K-Means Clustering:
The script defines a function `cluster` that takes a dataframe and the number of clusters (`n_clusters`) as inputs. It combines the processed numerical and binary features into a single dataframe and applies the K-Means clustering algorithm to this combined dataset. The function returns the fitted K-Means model, which categorizes the entries into different clusters based on their feature similarities.

### 3. Elbow Method for Optimal Clusters:
A function named `discover_optimal_k` is used to determine the optimal number of clusters for K-Means clustering. It calculates the "within-cluster sum of squares" (WCSS) for a range of cluster numbers (1 to 20) and plots these values to visually determine the elbow point. This point typically indicates the optimal cluster number, beyond which increasing the number of clusters does not significantly improve the clustering performance (diminishing returns).

### 4. Decision Tree Classifier:
The script also includes a `decision_tree` function that trains a Decision Tree Classifier on the same dataset. This classifier predicts the cluster labels (determined by the previously fitted K-Means model) for each entry based on the features. This approach might be used to understand or infer the rules that the clustering algorithm is implicitly using to segregate the data.

**Goals of the Code:**
- **Cluster Analysis:** Understand how fight songs can be grouped based on various attributes like beats per minute, duration, and thematic elements.
- **Optimal Clustering:** Determine the most effective number of clusters that summarize the data without overfitting or excessive granularity.
- **Predictive Modeling:** Develop a decision tree model that can predict cluster memberships, potentially offering insights into what features most strongly influence the grouping in fight songs.

Overall, this code aims to provide analytical insights into the dataset's structure through clustering, optimize the clustering process, and use a classification approach to predict and understand the cluster assignments. This can be particularly useful for data-driven decision-making in contexts where understanding categorical groupings within cultural or thematic datasets is valuable.

## Dataset
The dataset used in this project is named "fight-songs.csv". It includes both numerical features such as beats per minute (bpm) and duration in seconds, and binary features indicating various thematic elements like student writer presence, official song status, and others.

## Results
The clustering analysis reveals distinct groups of fight songs based on their features, providing insights into common themes and characteristics. The Decision Tree Classifier successfully predicts these cluster memberships, offering a deeper understanding of the factors that influence song categorization. The elbow method applied during the analysis helps in determining the optimal number of clusters, enhancing the model's effectiveness.
