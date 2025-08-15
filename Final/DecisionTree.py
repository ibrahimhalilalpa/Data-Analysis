from sklearn.cluster import Birch
import numpy as np

# Generating synthetic data: 10,000 customers with random spending and item count
np.random.seed(42)
data = np.random.rand(10000, 2) * [100, 10]  # Total spending (0-100) and item count (0-10)

# Initializing and fitting the BIRCH algorithm
birch_model = Birch(n_clusters=5)
birch_model.fit(data)

# Predicting clusters
birch_clusters = birch_model.predict(data)

# Example: Printing the first 10 cluster labels
print("BIRCH Clusters:", birch_clusters[:10])
