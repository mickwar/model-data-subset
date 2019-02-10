import numpy as np


np.random.seed(1)
n_vec = (500, 10000)
n_groups = 10

n = n_vec[0]
k = 5

x_array = np.zeros((n,k,n_groups))
y_array = np.zeros((n,1,n_groups))

### Form: group -- covariates -- response
# 1 -- standard normal -- sum of all covariates
x_array[:,:,0] = np.random.randn(n, k)
y_array[:,0,0] = np.sum(x_array[:,:,0], 1)

# 2 -- standard normal -- sum of all covariates
x_array[:,:,1] = np.random.randn(n, k)
y_array[:,0,1] = np.sum(x_array[:,:,1], 1)

# 3 -- standard normal -- sum of first two covariates
x_array[:,:,2] = np.random.randn(n, k)
y_array[:,0,2] = np.sum(x_array[:,:2,2], 1)

# 4 -- standard normal -- random normal
x_array[:,:,3] = np.random.randn(n, k)
y_array[:,0,3] = np.random.randn(n,)

# 5 -- high variance normal -- sum of all covariates
x_array[:,:,4] = np.random.randn(n, k) * 10
y_array[:,0,4] = np.sum(x_array[:,:,4], 1)

# 6 -- high variance normal -- sum of first two covariates
x_array[:,:,5] = np.random.randn(n, k) * 10
y_array[:,0,5] = np.sum(x_array[:,:2,5], 1)

# 7 -- high variance normal -- high variance random normal
x_array[:,:,6] = np.random.randn(n, k) * 10
y_array[:,0,6] = np.random.randn(n,) * 10

# 8 -- shifted standard normal -- sum of all covariates
x_array[:,:,7] = np.random.randn(n, k) + 5.0
y_array[:,0,7] = np.sum(x_array[:,:,7], 1)

# 9 -- shifted high variance normal -- sum of all covariates
x_array[:,:,8] = np.random.randn(n, k) * 10  + 5.0
y_array[:,0,8] = np.sum(x_array[:,:,8], 1)

# 10 -- shifted standard normal -- shifted normal
x_array[:,:,9] = np.random.randn(n, k) + 5.0
y_array[:,0,9] = np.random.randn(n,) + 5.0



