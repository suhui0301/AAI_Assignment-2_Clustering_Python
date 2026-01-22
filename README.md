# 🛒 Unsupervised Learning: Customer Segmentation (Clustering)

**Description:**
An Advanced Artificial Intelligence assignment focused on Unsupervised Machine Learning using K-Means Clustering. This project analyzes the [Wholesale Customers Dataset](https://archive.ics.uci.edu/dataset/292/wholesale+customers) sourced from the UCI Machine Learning Repository to identify distinct customer segments based on their annual spending behavior. Throught this assignment, I can perform knowledge discovery in finding hidden patterns in unlabelled data to inform business strategy without human supervision.z

---


## 👤 Student Details

| Name | Matric No. | Assignment Focus |
| --- | --- | --- |
| **Lau Su Hui (Abby)** | MEC245045 | Unsupervised Learning & K-Means Clustering (Python) |

---


## 📂 Project Modules

This repository contains the complete implementation and documentation for the Clustering assignment, focusing on the pipeline from data preprocessing to business insight extraction.

### 1. Source Code: MLP from Scratch
**File:** `UTM MECS1033 AAI Assignment 2 Clustering Code - Lau Su Hui MEC245045.py` in src folder
* **Dataset:** [UCI Wholesale Customers Data](https://archive.ics.uci.edu/dataset/292/wholesale+customers) (440 records, 6 features).
* **Technique:** K-Means Clustering with the Elbow Method for optimal _k_ selection.
* **Feature Engineering:** Implements `StandardScaler` to normalise high-variance data (e.g., preventing "Fresh" products from biasing the model).
* **Dimensionality Reduction:** Utilises Principal Component Analysis (PCA) to project 6-dimensional clusters onto a 2D plane for visualisation.


### 2. Project Report (Technical Analysis)
**File:** `UTM MECS1033 AAI Assignment 2 Clustering Report - Lau Su Hui MEC245045.pdf` in report folder
* **AI Methodology**: Comprehensive breakdown of the Unsupervised Learning pipeline, including data normalization, dimensionality reduction (PCA), and centroid initialization (k-means++).
* **Algorithm Evaluation**: Justification for the selected hyperparameter (k=3) using the Elbow Method heuristic to balance model complexity with Within-Cluster Sum of Squares (WCSS).
* **Knowledge Discovery**: Analysis of how the model extracted latent structures from unlabelled data, demonstrating the capability of clustering algorithms to find meaningful patterns without prior training labels.

---

## 🛠️ Technologies Used

* **Language:** Python
* **Core Concepts:** Unsupervised Learning, K-Means Clustering, Dimensionality Reduction (PCA), Data Preprocessing.
* **Libraries:**<br>
`panda`: For data manipulation and aggregation.<br>
`scikit-learn`: For KMeans, StandardScaler, and PCA algorithms.<br>
`matplotlib`: For generating the Elbow Curve and Cluster Scatter Plots.
`ucimlrepo`: For direct API access to the dataset.

---


## 🚀 How to Run

### Option 1: Google Colab (Recommended)
This is the easiest method as it requires no local setup.
1. Open Google Colab.
2. Create a New Notebook.
3. In the first code cell, install the required dataset library:
`!pip install ucimlrepo`
4. Copy and paste the full content of UTM MECS1033 AAI Assignment 2 Code - Lau Su Hui MEC245045.py into the next cell.
5. Press Run (Play Button).

Option 2: Local Execution
1. Ensure you have Python installed along with the required libraries: `pip install pandas scikit-learn matplotlib ucimlrepo`
2. Open your terminal and run: python "UTM MECS1033 AAI Assignment 2 Code - Lau Su Hui MEC245045.py"

---


📊 Understanding the Output

The program executes a complete Data Science pipeline:
* **Elbow Analysis**: Generates elbow_method_plot.png to visually justify the choice of k=3.
* **Clustering**: Assigns every customer to a specific group (0, 1, or 2).
* **Visualisation**: Saves `cluster_visualization.png` , showing the segments and centroids in PCA space.
* **Business Logic**: Prints a table of "Mean Spending" per cluster, revealing the "persona" of each group (e.g., who buys the most Detergents?).
* **Data Export**: Saves the final labelled dataset to `wholesale_customers_with_clusters.csv`.

---


## 📑 Project Insights

The project demonstrates how the **K-Means Clustering algorithm** can organise complex data without human supervision. Key Findings include:
* **Latent Structures:** The algorithm successfully identified 3 distinct groups:
1. Large Retailers (Cluster 0): High spend on Grocery & Detergents.
2. Small Standard Clients (Cluster 1): Moderate/Low spend across all categories (HoReCa).
3. Fresh Food Specialists (Cluster 2): Massive outliers in Fresh & Frozen produce.
* **Preprocessing Matters**: Standardising the data was critical; without it, the "Fresh" category (with values >100k) would have dominated the distance metric. This would render other features irrelevant.
* **PCA Visualisation**: Proved that while the data lives in 6 dimensions, the customer behaviours can be effectively mapped and separated in a 2-dimentional plane which is lower dimensional.
  
