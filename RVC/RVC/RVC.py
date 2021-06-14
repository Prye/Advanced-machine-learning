from functools import partial
import numpy as np
from scipy.optimize import minimize
from kernels import *

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_classifier(phi, w):
    return sigmoid(np.dot(phi, w))

def neg_log_p_wlt(w, target, phi, a):
    pred = sigmoid_classifier(phi, w)
    #entropy = target*np.log(pred, where=pred>0) + (1-target) * np.log(1-pred, where=(1-pred)>0)    # should avoid log(zero)
    positive_loc = np.where(target==1)[0]
    negative_loc = np.where(target==0)[0]
    entropy = np.sum( np.log(pred[positive_loc]))
    entropy += np.sum(np.log(1-pred[negative_loc]))
    tla = 0.5 * w.T.dot(a.dot(w))
    return -1*(entropy - tla)

def neg_gradient(w, target, phi, a):
    pred = sigmoid_classifier(phi, w)
    neg_gradient = a.dot(w) - phi.T.dot(target-pred)
    return neg_gradient

def get_laplace_approximation( w, alphas, phi, target):
    pred = sigmoid_classifier(phi, w)
    b = np.diag(pred*(1-pred))
    a = np.diag(alphas)
    # bishop 7.112
    w_star = np.linalg.inv(a).dot(phi.T.dot(target-pred))
    sigma = np.linalg.inv(phi.T.dot(b.dot(phi))+a)
    return w_star, sigma

    
def hessian(w, target, phi, a):
    pred = sigmoid_classifier(phi, w)
    b = np.diag(pred*(1-pred))
    # remove -1 since we are minimizing
    return phi.T.dot(b.dot(phi)) + a

class RVC:
    SUPPORTED_KERNEL = ['linear','poly','exp','rbf']
    def __init__(self, kernel, em_alpha_threshold=0.001, prune_threshold=1e9, poly_degree=None):
        self.em_alpha_th = em_alpha_threshold
        self.prune_th = prune_threshold
        self.__set_kernel(kernel, poly_degree)
        
        self.N = 0
        self.M = self.N+1
        # initial values
        self.alphas = []
        self.last_alphas = []

        self.relevance_vector = []

        self.has_bias = True

    def __set_kernel(self, kernel, poly_degree=None):
        if type(kernel) != str:
            self.kernel = kernel
        elif kernel in RVC.SUPPORTED_KERNEL:
            if kernel == 'linear':
                self.kernel = get_linear_kernel()
            elif kernel == 'rbf':
                self.kernel = get_rbf_kernel()
            elif kernel == 'poly':
                assert poly_degree is not None
                self.kernel = get_poly_kernel(poly_degree)
        else:
            raise ValueError('RVC kernel error')
        
    def fit(self, X, Y, max_it=50, initial_alpha=1e-6):
        self.N = X.shape[0]
        self.M = self.N + 1
        self.alphas = initial_alpha * np.ones(self.M)
        self.relevance_vector = X
        self.w0 = np.zeros(self.M)
        self.__run_em(max_it, Y)

    def predict(self, X):
        row_phi = self.kernel(X, self.relevance_vector)
        return sigmoid(row_phi.dot(self.w0[:-1])+self.w0[-1])
    
    def __run_em(self, max_iter, target):
        bias = np.ones((self.N,1))
        phi = self.kernel(self.relevance_vector,self.relevance_vector)
        phi = np.hstack((phi, bias))
        for it in range(max_iter):
            #w_star, sigma = self.__get_laplace_approximation(self.w0, self.alphas, phi, target)
            # https://tminka.github.io/papers/logreg/minka-logreg.pdf (2 Newton's method)
            # According to this, IRLS is equivalent to Newton's method
            irls = minimize(fun=neg_log_p_wlt, hess=hessian,
                            x0=self.w0, args=(target, phi, np.diag(self.alphas)),
                            method='Newton-CG', jac=neg_gradient,
                            options={'maxiter':50})
            w_star = irls.x
            sigma = np.linalg.inv(hessian(w_star, target, phi, np.diag(self.alphas)))
            # tipping: eq 9
            new_alphas = (1 - self.alphas*np.diag(sigma))/np.square(w_star)
            #print(new_alphas)
            diff = np.sum(np.abs(self.alphas - new_alphas))
            
            if diff < self.em_alpha_th:
                return
            else:
                # prune
                # avoid singular matrix error
                if False:
                    self.alphas = new_alphas
                    self.w0 = w_star
                else:
                    prune_loc = np.where(new_alphas < self.prune_th)[0]
                    self.alphas = new_alphas[prune_loc]
                    self.w0 = w_star[prune_loc]
                    phi = phi[:, prune_loc]
                    self.M = len(self.w0)
                    if self.has_bias:
                        self.relevance_vector = self.relevance_vector[prune_loc[:-1]]  # bias term            
                        #target = target[prune_loc[:-1]]
                        # prune bias term
                        if not prune_loc[-1]:
                            self.has_bias = False
                            self.N = self.M
                        else:
                            self.N = self.M-1
                    else:
                        self.N = self.M
                        self.relevance_vector = self.relevance_vector[prune_loc]  # bias term
                        #target = target[prune_loc]



if __name__ == "__main__":
    data_x = np.mgrid[-1:1:.1, -1:1:.1]
    data_x = data_x.reshape(2,-1).T
    N = data_x.shape[0]
    target = np.zeros(N)
    for n in range(N):
        target[n] = 1 if np.sin(data_x[n,0])<data_x[n,1] else 0
    rvc = RVC('linear')
    rvc.fit(data_x, target)

    test1 = [[0.3,2],[0.2,-1]]
    score = rvc.predict(test1)
    print(score)

        

    