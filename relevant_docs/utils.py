import hashlib
import numpy as np
import numba as nb
from tqdm.auto import tqdm
import pdb

def sort_entity_map(entity_map):
    entity_map = dict(entity_map)
    for e in tqdm(entity_map.keys()):
        entity_map[e] = np.sort(np.unique(entity_map[e]))
    return entity_map

@nb.njit
def intersect(l1, l2):
    i = j = k = 0
    ret = np.empty((min(l1.size, l2.size),))
    while i < l1.size and j < l2.size:
        if l1[i] == l2[j]:
            ret[k] = l1[i]
            i += 1
            j += 1
            k += 1
        elif l1[i] < l2[j]:
            i += 1
        else:
            j += 1
    return ret[:k]

@nb.njit
def difference(l1, l2):
    i = j = k = 0
    ret = np.empty((l1.size,))
    while i < l1.size:
        if j >= l2.size:
            ret[k] = l1[i]
            i += 1
            k += 1
        elif l1[i] == l2[j]:
            i += 1
            j += 1
        elif l1[i] < l2[j]:
            ret[k] = l1[i]
            i += 1
            k += 1
        else:
            j += 1
    return ret[:k]
