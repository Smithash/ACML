import math
import time
# Dataset consisting of datapoints in R^2
data = [(0.22, 0.33), (0.45, 0.76), (0.73, 0.39), (0.25, 0.35), (0.51, 0.69),
        (0.69, 0.42), (0.41, 0.49), (0.15, 0.29), (0.81, 0.32), (0.50, 0.88),
        (0.23, 0.31), (0.77, 0.30), (0.56, 0.75), (0.11, 0.38), (0.81, 0.33),
        (0.59, 0.77), (0.10, 0.89), (0.55, 0.09), (0.75, 0.35), (0.44, 0.55)]

# Number of clusters
k = 3

# Function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to assign data points to clusters
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(k)]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_idx = distances.index(min(distances))
        clusters[cluster_idx].append(point)
    return clusters

# Function to update cluster centroids
def update_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if cluster:
            x_sum = sum(point[0] for point in cluster)
            y_sum = sum(point[1] for point in cluster)
            centroid = (x_sum / len(cluster), y_sum / len(cluster))
            centroids.append(centroid)
    return centroids

# Read initial cluster centroids from input
stime= time.time()
centroids = []
for _ in range(3):
        x = float(input())
        y = float(input())
        centroids.append((x,y))

# Run k-means algorithm
prev_centroids = None
while centroids != prev_centroids:
    prev_centroids = centroids
    clusters = assign_clusters(data, centroids)
    centroids = update_centroids(clusters)

# Calculate sum-of-squares error
sum_of_squares_error = 0
for i, cluster in enumerate(clusters):
    for point in cluster:
        sum_of_squares_error += euclidean_distance(point, centroids[i])**2
etime = time.time()

# Print sum-of-squares error rounded to 4 decimal places
print(f"{sum_of_squares_error:.4f}")
print(etime - stime)