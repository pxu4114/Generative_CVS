import pickle 
import numpy as np
import pdb

file=np.load('/shared/kgcoe-research/mil/txt2img/flowers/test/label_test.npy')
file=list(file)
pdb.set_trace()
with open('/shared/kgcoe-research/mil/txt2img/flowers/test/label.pickle','w') as f:
	pick = pickle.Pickler(f)
	# for i in file:
	pick.dump(file)