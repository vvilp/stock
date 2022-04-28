import numpy as np

a = np.array([[1,2,3], [4,5,6], [5,6,7], [8,9,10]])
b = np.array([[2,3,4] , [6,7,9] , [6,7,8], [9,10,12]] )

print (np.square(a - b))
print (np.square(a - b).mean(1))
print (np.average(np.square(a - b).mean(1)))