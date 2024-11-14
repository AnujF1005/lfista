import matplotlib.pyplot as plt
import numpy as np
import os

def checkPatches(config):
    path = config["DATA_STORE_PATH"]
    path = os.path.join(path, "train")
    ps = config["PATCH_SIZE"]

    data = np.load(path + "/src/2.npy")

    for i in range(0, data.shape[0], 100):
        patch = data[i, :]
        patch = np.reshape(patch, (ps,ps))
        plt.imshow(patch, cmap='gray', vmin=0, vmax=1)
        plt.show()