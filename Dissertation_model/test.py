import numpy as np


aa = np.arange(24).reshape(2,4,3)

bb = np.mean(aa,-1)

print(aa)
print(bb)

