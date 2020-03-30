import numpy as np
from scipy import misc
import os

dataset = []
examples = []

data_root = "./data/"
alphabets = os.listdir(data_root)
for alphabet in alphabets:
    characters = os.listdir(os.path.join(data_root, alphabet))
    for img_file in characters:
        img = misc.imresize(
                            misc.imread(os.path.join(data_root, alphabet, img_file)), [32, 32, 3])
        examples.append(img)
    dataset.append(examples)
print (np.asarray(dataset).shape)
t = np.asarray(dataset)
t = t.reshape(t.shape[1], t.shape[0], t.shape[2], t.shape[3], t.shape[4])
print (t.shape)
np.save("./data_32_plant.npy", np.asarray(t))
