"""
clustering for 3-dimension continuous state to (10,3) discrete state 
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_state_distriburtion(data) :
    # model = cluster_load()
    model = KMeans(n_clusters=3, random_state=42)  
    model.fit(data)
    labels  = model.predict(data)
    centers = model.cluster_centers_
    dfData = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centers[:,0], centers[:,1], centers[:, 2], s = 250, marker='*', c='red', label='centroids')
    ax.scatter(dfData['X'], dfData['Y'], dfData['Z'], c=labels,s=40, cmap='winter')

    ax.set_title('K-Means Clustering')
    ax.set_xlabel('state : MSE')
    ax.set_ylabel('state : blur')
    ax.set_zlabel('state : network')
    ax.legend()
    # plt.show()
    plt.savefig('results/cluster.png')

    
def cluster_train(data, k=30):
    train_model = KMeans(n_clusters=k, random_state=42)  # else : MiniBatchKMeans
    train_model.fit(data)
    joblib.dump(train_model, '../models/cluster.pt')


def cluster_load():
    model = joblib.load("../models/cluster.pt")
    return model


def cluster_pred(sMSE, sFFT, sNet, model):
    env = [sMSE, sFFT, sNet]
    s = model.predict(env)
    return s

if __name__ == "__main__":
    data = []
    get_state_distriburtion(data)