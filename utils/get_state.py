"""
clustering for 3-dimension continuous state to (10,3) discrete state 
"""
import joblib
from sklearn import KMeans

def cluster_train(data, k=30) :
    model = KMeans(n_cluster=k)  # else : MiniBatchKMeans
    model.fit(data)
    joblib.dump(model, '../models/cluster.pt')
    
def get_state(sMSE, sFFT, sNet):
    env = [sMSE, sFFT, sNet]
    model = joblib.load("../models/cluster.pt")
    s = model.predict(env)
    return s