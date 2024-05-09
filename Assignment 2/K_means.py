import numpy as np

#Implementing K-means



#2. Dataset
data = np.array([[0.22, 0.33], [0.45, 0.76], [0.73, 0.39], [0.25, 0.35], 
                [0.51, 0.69], [0.69, 0.42], [0.41, 0.49], [0.15,0.29],
                [0.81, 0.32], [0.50, 0.88], [0.23, 0.31], [0.77, 0.30],
                [0.56, 0.75], [0.11, 0.38], [0.81, 0.33], [0.59, 0.77],
                [0.10, 0.89], [0.55, 0.09], [0.75, 0.35], [0.44, 0.55]])


#4. Run k-means Algorithm using hard coded dataset and starting eith cluster centres
    #5. halt when centres have converged
def K_Means(data, cluster_centers, K):
    initial_centres = cluster_centers
    
    while(True):
        
        #Compute the distance for each datapoint (datapoints - cluster_centers)
        #np.linalg.norm -> euclidean norm and saying axis =-1 compute the norm along last axis which calculates the euclidean distance
        distances = np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=-1)
        #assign x to its cluster where the distance is a minimum 
        assignments = np.argmin(distances, axis=1)

        #Update the centres
        new_centers = np.array([data[assignments == i].mean(axis=0) for i in range(K)])
        
        if np.all(cluster_centers == new_centers):
            break
        
        cluster_centers = new_centers
        
    return cluster_centers, assignments
        

    

#6. sum of squares error
def sum_of_squares_error(data, cluster_centres, assignments):
    sse = 0
    for i, centroid in enumerate(cluster_centres):
        cluster_points = data[np.where(assignments == i)]  # Get data points assigned to the current centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)  # Calculate distance to centroid
        sse += np.sum(distances ** 2)  # Sum of squared distances
    return np.round(sse,4) #round to 4 decimals

def main():
    #1. Number of clusters is 3
    K = 3
    
    #3. Read in from standard input a list of 6 numbers
    cluster_centers =[]
    for _ in range(3):
        x = float(input())
        y = float(input())
        cluster_centers.append((x,y))
    
    #4. K-Means
    final_cluster_centers, assignments = K_Means(data, cluster_centers, K)
    output = sum_of_squares_error(data, final_cluster_centers, assignments)
    print(output)
    
    
if __name__ == "__main__":
    main()