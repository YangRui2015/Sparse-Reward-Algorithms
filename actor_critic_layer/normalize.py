import numpy as np

class Normalizer:
    def __init__(self, size, eps=1e-2):
        self.size = size
        self.eps = eps
        self.sum_array = np.zeros(self.size, np.float32)
        self.sumsquare_array = np.zeros(self.size, np.float32)
        self.count = np.zeros(1)
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def reset(self):
        self.sum_array = np.zeros(self.size, np.float32)
        self.sumsquare_array = np.zeros(self.size, np.float32)
        self.count = np.zeros(1, np.float32)

        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def update(self, v):
        v = v.reshape(-1, self.size)
        self.sum_array += v.sum(axis=0)
        self.sumsquare_array += (np.square(v)).sum(axis=0)
        self.count += v.shape[0]
        assert self.count >= 1, "Count must be more than 1!"
        self.mean = self.sum_array / self.count
        self.std = np.sqrt(np.maximum(self.sumsquare_array / self.count - np.square(self.sum_array / self.count), np.square(self.eps)))

    def normalize(self, v):
        return (v - self.mean) / self.std


