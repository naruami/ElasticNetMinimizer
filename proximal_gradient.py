# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:42:13 2017

@author: kate
"""

import numpy as np
from pywt import threshold

#To speed up computations, I combine approx. error and L2 norm into the single formula
def compute_kernels(X,y,lambda2,weights=None):
    n_features=X.shape[1]
    #define weighted samples and regression matrix
    if np.any(weights):
        weights=np.diag(weights)
        WX=weights@X
        Wy=weights@y
    else:
        WX=X
        Wy=y
    
    #To speed up the computations further, we compute X^TW^2X and y^TWX (kernel trick)
    X_kernel=np.transpose(WX)@WX
    y_kernel=np.transpose(Wy)@WX
    

    #let us combine L2 norm of the features lambda2*w^Tw and w^TX^TXw into a single term
    X_kernel=X_kernel+lambda2*np.diag(np.ones(n_features))

    
    return X_kernel,y_kernel

#given the value of x, compute the value of the function, gradient, and the norm of the gradient   
def compute_values(X_kernel,y_kernel,w,lambda1):
    #We compute that part of the function that depends on features w, i.e.,
    #w^T*X_kernel*w
    func=(np.transpose(w)@X_kernel@w)[0,0]-2*(y_kernel@w)[0,0]+lambda1*np.linalg.norm(w,ord=1)
    grad=2*(np.transpose(w)@X_kernel)-2*y_kernel+lambda1*np.transpose(np.sign(w))
    #grad_norm=np.linalg.norm(grad)
    return func, grad#, grad_norm

def grad_norm_min(grad,w,lambda1):
    """
    Function that computes the least postential norm of subgradient
    Input: grad = gradient obtained by compute_values, 1-by-d np.matrix
            w = current value of the feature vector, d-by-1 matrix
            lambda1 = parameter lambda1 in Elastic Net, numeric
            weights1 = weights for weighted L1, d-dimensional array
    Output: least potential norm of the subgradient, numeric
    """
    grad_norm_squared = 0
    for i in range(w.shape[0]):
        if w[i,0]!=0:
            #derivative wrt this feature is well-defined
            grad_norm_squared += grad[0,i]**2
        elif np.abs(grad[0,i]) > lambda1:
            #then, the absolute value of the gradient can be reduced by weights1[i]
            grad_norm_squared += (np.abs(grad[0,i])-lambda1)**2
            #if w[i,0]==0 and np.abs(grad[0,i])<=weights1[i], then the subgradient wrt
            #i-th feature can be chosen to be 0
    return np.sqrt(grad_norm_squared)

    


def proximal_gradient_descent_constant_stepsize(
    X_kernel,y_kernel,w,lambda1=1,stepsize=-1, maxIter=100000,precision=1e-10):
    """
    Name: proximal_gradient_descent_constant_stepsize
    Goal: compute the minimum of the function
        w^T X_kernel w - 2 y_kernel w +lambda1||w||_1
        with respect to w. The minumum is found using the proximal gradient
        descent with constant stepsize.
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial input value
        lambda1 = numeric, coefficient for L1 norm
        stepsize = numeric, value of the stepsize. Default =-1, i.e.,
            will be computed based on eigenvalues of matrix X_kernel
        maxIter = integer, maximum number of iterations allowed, default =100000
        precision = numeric, if least possible subgradient norm is less, we stop
    Output:
        w = d-by-1 np.matrix, features when the value of the function is the least, first entry is bias
        func = numeric, least possible value of function
        grad_norm = minimum norm of the subgradient
        converged = boolean, indicates if the method has converged
    """
    func, grad = compute_values(X_kernel,y_kernel,w,lambda1)
    grad_norm = grad_norm_min(grad,w,lambda1)
    iteration=0
    precision=precision*np.sqrt(len(w))#make precision proportional to the dimension of the gradient   
    print("-----------------------------------------------------")
    print("Proximal gradient descent with constant stepsize is initiated.")
    print("-----------------------------------------------------")
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    #To implement the algorithm with constant stepsize, recall that
    #stepsize=1/sigma, where
    #sigma I~ Hessian of (w^T X_kernel w-2y_kernel w)
    #However, the Hessian of the last expression can be computed analytically
    #and equals 2*X_kernel
    #I approximate Hessian of X_kernel as average between the largest and the smallest eigenvalue
    #and multiply it by 2
    if stepsize==-1:
        sigma=(np.linalg.norm(X_kernel,ord=2)+np.linalg.norm(X_kernel,ord=-2))
        stepsize=1/sigma
    change_w=100
    while ((grad_norm>precision)&(iteration<maxIter)&(change_w>1e-10)):
        iteration+= 1
        threshold_value = lambda1*stepsize
        w_new = threshold(w-stepsize*np.transpose(grad),threshold_value,mode="soft")
        func_new,grad_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
        change_w=np.linalg.norm(w-w_new)
        w = w_new.copy()
        func = func_new.copy()
        grad = grad_new.copy()
        grad_norm = grad_norm_min(grad,w,lambda1)
        #print("iter=%s||norm(x)=%s||f_cur=%s||grad=%s"%(iteration,np.linalg.norm(x),func,grad_norm))
        
        if iteration in range(0,maxIter,20000):
            print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    if grad_norm<precision:
        print("The proximal gradient descent converged after %s iterations." %iteration)
        converged = True
    elif change_w<=1e-10:
        print("The proximal gradient descent converged numerically, current change in w is too small (<1e-10)")
        converged = True
    else:
        print("The proximal gradient descent did not converge.")
        converged = False
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    print("-----------------------------------------------------")
    return w, func,grad_norm, converged

def proximal_gradient_descent_BB_stepsize(
    X_kernel,y_kernel,w,lambda1=1, stepsize=-1,maxIter=100000,precision=1e-10):
    """
    Name: proximal_gradient_descent_BB_stepsize
    Goal: compute the minimum of the function
        w^T X_kernel w - 2 y_kernel w +lambda1||w||_1
        with respect to w. The minumum is found using the proximal gradient
        descent with Borzelai-Borwein stepsize.
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial input value
        lambda1 = numeric, coefficient for L1 norm
        stepsize = numeric, value of the stepsize. Default =-1, i.e.,
            will be computed based on eigenvalues of matrix X_kernel
        maxIter = integer, maximum number of iterations allowed, default =100000
        precision = numeric, if least possible subgradient norm is less, we stop,default=1e-10
    Output:
        w = d-by-1 np.matrix, features when the value of the function is the least, first entry is bias
        func = numeric, least possible value of function
        grad_norm = minimum norm of the subgradient
        converged = boolean, indicates if the method has converged
    """
    func, grad = compute_values(X_kernel,y_kernel,w,lambda1)
    grad_norm = grad_norm_min(grad,w,lambda1)
    iteration=0
    precision=precision*np.sqrt(len(w))#make precision proportional to the dimension of the gradient   
    print("-----------------------------------------------------")
    print("Proximal gradient descent with Barzilai-Borwein stepsize is initiated.")
    print("-----------------------------------------------------")
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    #To implement the algorithm with Barzilai-Borwein(BB) stepsize
    #stepsize=||change_w||^2/(change_grad*change_w)
    #where change_w = w-w_prev
    #change_grad = grad- grad_prev
    
    #We start with simple iteration to get initial change_w and change_grad
    iteration+= 1
    if stepsize==-1:
        sigma=(np.linalg.norm(X_kernel,ord=2)+np.linalg.norm(X_kernel,ord=-2))
        stepsize=1/sigma
    threshold_value = lambda1*stepsize
    w_new = threshold(w-stepsize*np.transpose(grad),threshold_value,mode="soft")
    func_new,grad_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
    change_w = w_new-w
    change_grad = grad_new - grad
    #Update the values of feature vector, function value, gradient, and gradient norm
    w = w_new.copy()
    func = func_new.copy()
    grad = grad_new.copy()
    grad_norm = grad_norm_min(grad,w,lambda1)
        
    #Now we are ready to implement the proximal gradient with BB stepsize
    while ((grad_norm>precision)&(iteration<maxIter)&(np.linalg.norm(change_w)>1e-10)):
        iteration+= 1
        stepsize = (np.linalg.norm(change_w))**2/np.dot(change_grad,change_w)[0,0]
        threshold_value = lambda1*stepsize
        w_new = threshold(w-stepsize*np.transpose(grad),threshold_value,mode="soft")
        func_new,grad_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
        #Update change_w and change_grad
        change_w = w_new-w
        change_grad = grad_new-grad
        #Assign current value of function new values
        w = w_new.copy()
        func = func_new.copy()
        grad = grad_new.copy()
        grad_norm = grad_norm_min(grad,w,lambda1)
        #print("iter=%s||norm(x)=%s||f_cur=%s||grad=%s"%(iteration,np.linalg.norm(x),func,grad_norm))
        
        if iteration in range(0,maxIter,20000):
            print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    if grad_norm<precision:
        print("The proximal gradient descent converged after %s iterations." %iteration)
        converged = True
    elif np.linalg.norm(change_w)<=1e-10:
        print("The proximal gradient descent converged numerically, current change in w is too small (<1e-10)")
        converged = True
    else:
        print("The proximal gradient descent did not converge.")
        converged = False
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    print("-----------------------------------------------------")
    return w,func,grad_norm,converged


    
def _test(n_features=10,n_samples=101,warm_start=True):
    indexes=np.random.randint(0,n_features,3)
    true_w =np.zeros((n_features+1,1))
    true_w[indexes,:]=np.matrix(np.random.randn(3,1))
    true_w[0,:] = np.random.randn(1)
    X = np.random.randn(n_samples,n_features+1)
    X[:,0]=np.ones(n_samples)

    error=np.random.randn(n_samples,1)
    y=X@true_w+error
    
    lambda1=3
    lambda2=1
    X_kernel,y_kernel = compute_kernels(X,y,lambda2)
    if warm_start==False:
        x = np.random.randn(n_features+1,1)
    else:
        #warm start
        x = np.linalg.pinv(X_kernel)@np.transpose(y_kernel)

    x = np.linalg.pinv(X_kernel)@np.transpose(y_kernel)
    print("Initial norm of error %s"%np.linalg.norm(true_w-x))
    

    x_min,f_min,converged = proximal_gradient_descent_constant_stepsize(X_kernel,y_kernel,x,lambda1=lambda1)
    print("Norm of error %s"%np.linalg.norm(true_w-x_min))
    
    x_min,f_min,converged =proximal_gradient_descent_BB_stepsize(X_kernel,y_kernel,x,lambda1=lambda1)
    print("Norm of error %s"%np.linalg.norm(true_w-x_min))
    print("Initial norm of error %s"%np.linalg.norm(true_w-x))

        
