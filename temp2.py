import numpy as np
from pycv._lib._src import c_pycv


########################################################################################################################

inputs = np.array(
    [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
     [0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 1],
     [0, 0, 0, 2, 2, 2, 2, 0, 0, 1, 0],
     [0, 0, 0, 0, 2, 2, 2, 0, 0, 1, 0],
     [0, 1, 1, 1, 0, 2, 2, 1, 1, 0, 0],
     [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    dtype=np.uint8
)

ff = inputs.shape[1]

ind = np.arange(inputs.size, dtype=np.int64).reshape(inputs.shape)

labels = np.empty((inputs.size, ), dtype=np.int64)
pp = 0
nl = 0

for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        if not inputs[i, j]:
            labels[pp] = -1
            pp += 1
            continue
        labels[pp] = pp
        edges = []
        for cc, nn in [((i - 1, j), pp - ff), ((i, j - 1), pp - 1), ((i - 1, j - 1), pp - ff - 1), ((i - 1, j + 1), pp - ff + 1)]:
            if any(c < 0 for c in cc) or any(cc[n] >= inputs.shape[n] for n in range(2)) or inputs[cc] != inputs[i, j]:
                continue
            edges.append(nn)
            labels[pp] = min(labels[pp], labels[nn])

        if not edges:
            nl += 1
            pp += 1
            continue

        p = labels[pp]
        while labels[p] != p:
            p = labels[p]

        for edge in edges:
            e = edge
            while labels[e] != e:
                e = labels[e]

            if e != p:
                nl -= 1
                labels[e] = p
        pp += 1

out = np.zeros((inputs.size, ), dtype=np.int64)
nn = 1

for i in range(labels.size):
    if labels[i] != labels[labels[i]]:
        labels[i] = labels[labels[i]]

    if labels[i] != -1 and out[labels[i]] == 0:
        out[labels[i]] = nn
        nn += 1

    out[i] = out[labels[i]]



o = out.reshape(inputs.shape)