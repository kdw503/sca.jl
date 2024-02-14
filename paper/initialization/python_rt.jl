
# now python

# activate environment at the power shell prompt
Scripts\activate.bat
# To deactivate a virtual environment, type:
# deactivate

# launch python
py

#
import time
import numpy as np
from numpy import random
from sklearn.utils.extmath import randomized_svd, squared_norm, randomized_range_finder

nr = 15
r_ov = 10


rt0 = time.time()
L = randomized_range_finder(X, size = nr + r_ov, n_iter = 3)
R = randomized_range_finder(X.T, size = nr + r_ov, n_iter = 3)
rt = time.time()-rt0

np.savez("C:\\Users\\kdw76\\WUSTL\\Work\\julia\\sca\\paper\\compnmf\\LR-10dB_factor1.npz",L,R)


for f in range()
    X = random.randint(100, size=(3, 5))
    rt = 0
    for i in range(1,50):
    #    print(i)
        rt0 = time.time()
        L = randomized_range_finder(X, size = nr + r_ov, n_iter = 3)
        R = randomized_range_finder(X.T, size = nr + r_ov, n_iter = 3)
        rt += time.time()-rt0

rt/50


# julia again
using NPZ
LR = npzread(joinpath(subworkpath,"LR-10dB_factor1.npz"))
print(LR.files)
L = LR["arr_0"]
R = LR["arr_1"]'
