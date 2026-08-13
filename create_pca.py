import pickle
import numpy as np
from sklearn.decomposition import PCA
from utils.mix_loader import MixDataset

dataset = MixDataset("train")

X = []

for feat, _ in dataset:
    X.append(feat.numpy())

X = np.concatenate(X, axis=0)

print(X.shape)

pca = PCA(
    n_components=512
)

pca.fit(X)

with open(
    "checkpoints/pca_mix.pkl",
    "wb"
) as f:

    pickle.dump(pca, f)

print("Done")