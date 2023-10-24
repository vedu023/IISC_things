import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from skimage import io

def calculate_similarity_matrix(image):
    height, width, _ = image.shape
    num_pixels = height * width 
    similarity_matrix = sp.lil_matrix((num_pixels, num_pixels), dtype=np.float32) 
    flattened_image = image.reshape((num_pixels, -1))
 
    for i in range(num_pixels):
        for j in range(i, num_pixels):
            distance = np.linalg.norm(flattened_image[i] - flattened_image[j])
            similarity_matrix[i, j] = similarity_matrix[j, i] = np.exp(-distance) 
    similarity_matrix = similarity_matrix.tocsr()

    return similarity_matrix

def ncut_segmentation(image, k):
     
    image = io.imread(image)
    height, width, _ = image.shape
    num_pixels = height * width

    def create_graph(image):
        similarity_matrix = calculate_similarity_matrix(image) 
        degree_matrix = np.diag(similarity_matrix.sum(axis=1))
        laplacian_matrix = degree_matrix - similarity_matrix

        return laplacian_matrix

    laplacian_matrix = create_graph(image) 
    num_eigenvalues = k + 1   
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=num_eigenvalues, which='SM')

    # Cluster the eigenvectors (e.g., using K-means)
    kmeans = KMeans(n_clusters=k)
    eigenvectors = eigenvectors[:, 1:]  # Exclude the first eigenvector
    segment_labels = kmeans.fit_predict(eigenvectors)
 
    segmentation = segment_labels.reshape((height, width))

    return segmentation

path = 'AIP_assignments/a1/second.jpeg'
segmented_image = ncut_segmentation(path, k=2)
io.imshow(segmented_image)
io.show()
