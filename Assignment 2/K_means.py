import numpy as np
import time
import math

#Using numoy is signifcantly slower than just using math library
# 2. Dataset
data = [[0.22, 0.33], [0.45, 0.76], [0.73, 0.39], [0.25, 0.35], [0.51, 0.69],
                 [0.69, 0.42], [0.41, 0.49], [0.15, 0.29], [0.81, 0.32], [0.50, 0.88],
                 [0.23, 0.31], [0.77, 0.30], [0.56, 0.75], [0.11, 0.38], [0.81, 0.33],
                 [0.59, 0.77], [0.10, 0.89], [0.55, 0.09], [0.75, 0.35], [0.44, 0.55]]

def e_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def assign_clusters(data, cluster_centers, K):
    assign = [[] for _ in range(K)]
    for d in data:
        distance = [e_distance(d, cluster_center) for cluster_center in cluster_centers]
        cluster_idx = distance.index(min(distance))
        assign[cluster_idx].append(d)
    return assign
# 4. Run k-means Algorithm using hard coded dataset and starting with cluster centers
# 5. Halt when centers have converged
def K_Means(data, initial_centers, K):
    cluster_centers = initial_centers
    prev_centers = None
    while prev_centers != cluster_centers:
        prev_centers = cluster_centers
        # Compute squared distances and assign data points to clusters
        assignments = assign_clusters(data, cluster_centers, K)
        
        # Update centers
        new_centers =[]
        for a in assignments:
            if a:
                x_sum = sum(point[0] for point in a)
                y_sum = sum(point[1] for point in a)
                new_center = (x_sum/len(a), y_sum/len(a))
                new_centers.append(new_center)

        # Check if centers have converged
        
        cluster_centers = new_centers

    return cluster_centers, assignments

# 6. Sum of squares error
def sum_of_squares_error(data, cluster_centers, assignments):
    sum_of_squares_error = 0
    for i, cluster in enumerate(assignments):
        for point in cluster:
            sum_of_squares_error += e_distance(point, cluster_centers[i])**2
    print(f"{sum_of_squares_error:.4f}")

def main():
    # 1. Number of clusters is 3
    K = 3

    # 3. Read in from standard input a list of 6 numbers
    cluster_centers = []
    
    for _ in range(3):
        x = float(input())
        y = float(input())
        cluster_centers.append((x, y))

    # 4. K-Means
    final_cluster_centers, assignments = K_Means(data, cluster_centers, K)
    sum_of_squares_error(data, final_cluster_centers, assignments)
    
    
   

if __name__ == "__main__":
    main()