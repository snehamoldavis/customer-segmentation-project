import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# Sample Customer Data
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Age": [25, 34, 45, 23, 40, 50, 22, 36, 48, 29],
    "AnnualIncome_k": [15, 16, 48, 48, 54, 70, 81, 85, 100, 120],
    "SpendingScore": [39, 81, 6, 77, 40, 55, 93, 39, 42, 90],
}
df = pd.DataFrame(data)

# Features for clustering (Income and Spending Score)
X = df[["AnnualIncome_k", "SpendingScore"]]

# Apply K-Means Clustering (Dividing into 3 customer groups)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# Print the segmented data in terminal
print(df)

# Visualize and save the segments as an image
plt.figure(figsize=(8, 6))
plt.scatter(
    df["AnnualIncome_k"],
    df["SpendingScore"],
    c=df["Cluster"],
    cmap="viridis",
    s=150,
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income ($k)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.savefig("customer_segments.png")
print("Graph saved successfully as customer_segments.png!")
plt.show()