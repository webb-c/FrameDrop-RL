"""
clustering for 3-dimension continuous state to (10,3) discrete state 
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_state_distriburtion(model, data) :
    labels  = model.predict(data)
    centers = model.cluster_centers_
    dfData = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centers[:,0], centers[:,1], centers[:, 2], s =250, marker='*', c='red', label='centroids')
    ax.scatter(dfData['X'], dfData['Y'], dfData['Z'], c=labels,s=40, cmap='winter')

    ax.set_title('K-Means Clustering')
    ax.set_xlabel('state : MSE')
    ax.set_ylabel('state : blur')
    ax.set_zlabel('state : network')
    ax.legend()
    # plt.show()
    plt.savefig('results/cluster_(1_1).png')

def cluster_init(k=10):
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    return model
    
def cluster_train(model, data, clusterPath, visualize=False):
    model.fit(data)
    joblib.dump(model, clusterPath)
    if visualize :
        get_state_distriburtion(model, data)
    return model


def cluster_load(clusterPath):
    model = joblib.load(clusterPath)
    return model


def cluster_pred(originState, model):
    originState = [originState]
    s = model.predict(originState)
    return s

if __name__ == "__main__":
    data = []
    get_state_distriburtion(data)