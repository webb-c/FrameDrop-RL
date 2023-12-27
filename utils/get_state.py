"""
clustering for 3-dimension continuous state to (10,3) discrete state 
"""
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def get_state_distriburtion(model, data, clusterPath) :
    labels  = model.predict(data)
    centers = model.cluster_centers_
    dfData = pd.DataFrame(data, columns=['X', 'Y'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(centers[:,0], centers[:,1], s =250, marker='*', c='red', label='centroids')
    ax.scatter(dfData['X'], dfData['Y'], c=labels,s=40, cmap='winter')

    ax.set_title('K-Means Clustering')
    ax.set_xlabel('state : MSE')
    ax.set_ylabel('state : blur')
    ax.legend()
    # plt.show()
    video_name = clusterPath.split("/")[-1].split(".")[0]
    path = "results/figure/"+video_name+".png"
    plt.savefig(path)

def cluster_init(state_num=15):
    model = KMeans(n_clusters=state_num, n_init=10, random_state=42)
    return model
    
def cluster_train(model, data, clusterPath, visualize=False):
    print("start clustering for inputVideo")
    model.fit(data)
    joblib.dump(model, clusterPath)
    if visualize :
        get_state_distriburtion(model, data, clusterPath)
    return model


def cluster_load(clusterPath):
    model = joblib.load(clusterPath)
    return model


def cluster_pred(originState, model):
    originState = [originState]
    s = model.predict(originState)
    return s
