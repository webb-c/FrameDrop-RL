"""
clustering for 3-dimension continuous state to (10,3) discrete state 
"""
import joblib
from sklearn.cluster import KMeans


def cluster_train(data, k=30):
    train_model = KMeans(n_cluster=k)  # else : MiniBatchKMeans
    train_model.fit(data)
    joblib.dump(train_model, '../models/cluster.pt')


def cluster_load():
    model = joblib.load("../models/cluster.pt")
    return model


def cluster_pred(sMSE, sFFT, sNet, model):
    env = [sMSE, sFFT, sNet]
    s = model.predict(env)
    return s
