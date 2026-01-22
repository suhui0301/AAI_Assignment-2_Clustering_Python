#1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ucimlrepo import fetch_ucirepo

# Compatibility for running in standard Python script vs IPython/Jupyter
try:
    from IPython.display import display
except ImportError:
    display = print

#2. DATA LOADING
print("Step 1: Loading Dataset...")
# Fetch dataset using the ID 292 for Wholesale Customers
wholesale_customers = fetch_ucirepo(id=292)

# Get features and targets
features_all = wholesale_customers.data.features # Contains Channel + 6 Products
targets_original = wholesale_customers.data.targets # Contains Region

# Define X (Clustering Features) and y (Categorical Labels for Analysis)
# Standard: X = Continuous Spending Features, y = Categorical Labels (Channel, Region)
X = features_all.drop(columns=['Channel']) # Drop Channel from X
y = targets_original.copy()
y['Channel'] = features_all['Channel'] # Add Channel to y

# Combine everything into one DataFrame for inspection/saving
full_data = X.copy()
full_data = pd.concat([full_data, y], axis=1)

# Save the raw dataset
full_data.to_csv('wholesale_customers_raw.csv', index=False)
print("Raw dataset saved to 'wholesale_customers_raw.csv'")

print("Dataset loaded successfully!")
print(f"X shape (Clustering Features): {X.shape}")
print(f"y shape (Categorical Labels): {y.shape}")
print("Features used for clustering:", X.columns.tolist())
print("Labels available for analysis:", y.columns.tolist())
print("-" * 30)

#3. DATA EXPLORATION (EDA)
print("Step 2: Exploring Data...")
# Show first few rows
print("First 5 rows of data:")
display(X.head())

# Show basic statistics
print("\nData Statistics:")
display(X.describe())
print("-" * 30)

#4. PREPROCESSING
print("Step 3: Preprocessing (Scaling)...")
# K-Means is sensitive to scale, so we must standardize the data (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for easier handling later (optional but good for debugging)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("Data scaled.")
print("-" * 30)

#5. DETERMINING OPTIMAL K (ELBOW METHOD)
print("Step 4: Running Elbow Method...")
wcss = [] # Within-Cluster Sum of Squares

# Try k from 1 to 10
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal K')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Distortion)')
plt.grid(True)
plt.savefig('elbow_method_plot.png') # Save plot to file
print("Elbow plot saved as 'elbow_method_plot.png'. Check this image to choose K.")
print("-" * 30)

# 6. VALIDATION (SILHOUETTE ANALYSIS)
from sklearn.metrics import silhouette_score

print(f"Step 5: Running Silhouette Analysis...")
silhouette_scores = []

# Try k from 2 to 10 (Silhouette score requires at least 2 clusters)
for k in range(2, 11):
    kmeans_sil = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans_sil.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}, Silhouette Score={score:.4f}")

# Plot Silhouette Graph
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='green')
plt.title('Silhouette Scores for Various k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Coefficient')
plt.grid(True)
plt.savefig('silhouette_plot.png')
print("Silhouette plot saved as 'silhouette_plot.png'.")
print("-" * 30)

# 7. APPLYING K-MEANS
# Based on common patterns in this dataset, k=3 or k=5 is usually good.
# Use k=3 for this demonstration as it's easier to explain.
optimal_k = 3 
print(f"Step 6: Applying K-Means with k={optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original full data
X_with_clusters = full_data.copy()
X_with_clusters['Cluster'] = clusters

print("Clustering complete.")
print("-" * 30)

# 8. VISUALIZATION (PCA)
print("Step 7: Visualizing Clusters (2D PCA)...")
# Reduce 6 dimensions (features) to 2 dimensions for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
# Scatter plot for each cluster
for i in range(optimal_k):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i}')

# Plot the Centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', marker='X', label='Centroids', edgecolors='black')

plt.title(f'Customer Segments (K-Means k={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig('cluster_visualization.png') # Save plot to file
print("Cluster visualization saved as 'cluster_visualization.png'.")
print("-" * 30)

# 9. ANALYSIS & INTERPRETATION
print("Step 8: Cluster Analysis (Mean Spending)...")
# Group by cluster and calculate mean spending for each category
cluster_means = X_with_clusters.groupby('Cluster').mean()
print(cluster_means)

# Save the results to a CSV
X_with_clusters.to_csv('wholesale_customers_with_clusters.csv', index=False)
print("\nDetailed results saved to 'wholesale_customers_with_clusters.csv'.")
print("Done!")
