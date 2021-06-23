import numpy as np

arrs=[np.array([[1,2,3,4],[5,6,7,9]]),np.array([[1,2,3,4],[5,6,8,9]]),np.array([[1,2,3,4],[5,6,7,8]]),np.array([[1,2,3,4],[5,6,7,8]])]
new_arrs=np.dstack(arrs)
final_arrs=np.rollaxis(new_arrs,-1)