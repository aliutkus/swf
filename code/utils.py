import numpy as np


def compare_image(image, collection, return_amount):
    lowest_mse = 1000000000
    ax = None
    arr = np.array([]).astype(np.float32)
    for i in range(len(collection)):
        mse = ((image.detach().numpy() - collection[i][0].detach().numpy()) ** 2).mean(axis=ax)
        arr = np.append(arr, mse)
        if mse < lowest_mse:
            lowest_mse = mse
    ind = np.argpartition(arr, return_amount)[:return_amount]
    return ind, arr