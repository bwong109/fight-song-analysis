import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt

numerical_features = [
    'bpm', 'sec_duration', 'number_fights', 'trope_count']

binary_features = [
    'student_writer', 'official_song', 'contest', 'fight',
    'victory', 'win_won', 'victory_win_won', 'rah',
    'nonsense', 'colors', 'men', 'opponents', 'spelling']

from sklearn.preprocessing import StandardScaler

def cluster(df, n_clusters):
    numerical = df[numerical_features]
    binary = df[binary_features].replace({'Yes': 1, 'No': 0, 'Unknown': 0})

    binary = StandardScaler().fit_transform(binary)
    numerical = StandardScaler().fit_transform(numerical)

    X = pd.concat([pd.DataFrame(numerical, columns=numerical_features), pd.DataFrame(binary, columns=binary_features)],
                  axis=1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)

    return kmeans

df = pd.read_csv("fight-songs.csv")
clustering = cluster(df, 5)

def discover_optimal_k(df):
    numerical = df[numerical_features]
    binary = df[binary_features].replace({'Yes': 1, 'No': 0, 'Unknown': 0})

    binary = StandardScaler().fit_transform(binary)
    numerical = StandardScaler().fit_transform(numerical)

    X = pd.concat([pd.DataFrame(numerical, columns=numerical_features), pd.DataFrame(binary, columns=binary_features)],
                  axis=1)

    errors = []

    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        errors.append(kmeans.inertia_)

    plt.plot(range(1, 21), errors)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans Clustering')

    return np.array(errors)

discover_optimal_k(df);


def decision_tree(df, kmeans):
    numerical = df[numerical_features]
    binary = df[binary_features].replace({'Yes': 1, 'No': 0, 'Unknown': 0})

    binary = StandardScaler().fit_transform(binary)
    numerical = StandardScaler().fit_transform(numerical)

    X = pd.concat([pd.DataFrame(numerical, columns=numerical_features), pd.DataFrame(binary, columns=binary_features)],
                  axis=1)

    cluster_labels = kmeans.predict(X)
    decisionTree = DecisionTreeClassifier(random_state=0)
    decisionTree.fit(X, cluster_labels)

    return decisionTree

