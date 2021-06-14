import numpy as np
import sklearn.gaussian_process.kernels as gpk
def get_rbf_kernel():
    return gpk.RBF()
def get_linear_kernel():
    return gpk.DotProduct(sigma_0=0)
def get_poly_kernel(exp):
    return gpk.Exponentiation(get_linear_kernel,exp)