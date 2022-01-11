import numpy as np
from sklearn.cluster import KMeans


class SpectralClustering():
    labels = []

    def __init__(self, n_clusters, gamma=50, n_knn=10):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.n_knn = n_knn

    def calc_euclidean_distance(self, x1, x2, sqrt_flag=False):
        res = np.sum((x1 - x2) ** 2)
        if sqrt_flag:
            res = np.sqrt(res)
        return res

    def calc_euclidean_distance_matrix(self, X):
        X = np.array(X)
        S = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                S[i][j] = 1.0 * self.calc_euclidean_distance(X[i], X[j])
                S[j][i] = S[i][j]
        return S

    def get_affinity_matrix(self, data, ):
        S = self.calc_euclidean_distance_matrix(data)

        N = len(S)
        A = np.zeros((N, N))

        # KNN
        for i in range(N):
            dist_with_index = zip(S[i], range(N))
            dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
            neighbours_id = [dist_with_index[m][1] for m in range(self.n_knn + 1)]  # xi's k nearest neighbours

            for j in neighbours_id:  # xj is xi's neighbour
                A[i][j] = np.exp(-2 * S[i][j] * self.gamma)
                A[j][i] = A[i][j]  # mutually

        return A

    def calc_laplacian_matrix(self, affinity_matrix):
        # compute the Degree Matrix: D=sum(A)
        degree_matrix = np.sum(affinity_matrix, axis=1)

        # compute the Laplacian Matrix: L=D-A
        laplacian_matrix = np.diag(degree_matrix) - affinity_matrix

        # normailze: D^(-1/2) L D^(-1/2)
        sqrt_degree_matrix = np.diag(1.0 / (degree_matrix ** (0.5)))
        return np.dot(np.dot(sqrt_degree_matrix, laplacian_matrix), sqrt_degree_matrix)

    def eig(self, L):
        x, V = np.linalg.eig(L)
        return x, V

    def fit(self, data):
        W = self.get_affinity_matrix(data)
        L = self.calc_laplacian_matrix(W)
        x, V = self.eig(L)
        x = zip(x[:], range(len(x[:])))
        x = sorted(x, key=lambda x: x[0])
        H = np.vstack([V[:, i] for (v, i) in x[:data.shape[0]]]).T
        kmeans = KMeans(n_clusters=self.n_clusters).fit(H)
        self.labels_ = kmeans.labels_
        return self


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from itertools import cycle, islice
    from sklearn import datasets


    def make_circles(n_samples=1000):
        X, y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
        return X, y


    def plot(X, y_sp, y_km):
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_km) + 1))))
        plt.subplot(121)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_sp])
        plt.title("Spectral Clustering")
        plt.subplot(122)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_km])
        plt.title("Kmeans Clustering")
        # plt.show()
        plt.savefig("spectral_clustering.png")


    data, label = make_circles(n_samples=500)
    sc = SpectralClustering(n_clusters=2, gamma = 60).fit(data)
    pure_kmeans = KMeans(n_clusters=2).fit(data)
    plot(data, sc.labels_, pure_kmeans.labels_)
