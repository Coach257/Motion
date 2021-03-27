import os
import pickle as pk
import json
import numpy as np

"""
Schedule cache for data loading
"""
class CacheSchedule(object):
    def __init__(self, cache_size, data_path, files, type = "pkl"):
        self.cache_size = cache_size
        if(type == "pkl"):
            self.cache = [pk.load(open(os.path.join(data_path, f), 'rb')) for f in files]
        else:
            assert type == "json"
            self.cache = [json.load(open(os.path.join(data_path, f), 'rb')) for f in files]
        print("Cache schedule num:", len(self.cache))

    def load(self, file, index):
        # if file in self.cache:
        #     return self.cache[file]
        # else:
        #     data = pk.load(open(file, "rb"))
        #     self.cache[file] = data
        #     return data
        return self.cache[index]
        # return pk.load(open(file, 'rb'))