# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:35:47 2017

@author: Kateryna Melnykova
"""

import numpy as np


#To speed up computations, I combine approx. error and L2 norm into the single formula
def compute_kernels(X,y,lambda2,weights=None):
    """
    This function is for internal computing only. It's main goal is to speed
    up computations of the target function
    Input:
        X = N-by-d np.matrix, regression matrix, bias has been added already if needed
        y = N-by-1 np.matrix, output variable
        lambda2 = weights for l2 norm, can be numeric, d-by-1 np.matrix,
                    or d-dimensional np.array, or d-by-1 np.array
    Output:
        X_kernel = kernel matrix for combined residuals and l2 norm, equals
                    X^T W^2 X +lambda2
        y_kernel = kernel vector, equals y^T W^2 X 
    """
    n_features = X.shape[1]
    #define weighted samples and regression matrix
    if np.any(weights):
        weights=np.diagflat(weights)
        WX = weights@X
        Wy = weights@y
    else:
        WX = X
        Wy = y
    
    #To speed up the computations further, we compute X^TW^2X and y^TWX (kernel trick)
    X_kernel=np.transpose(WX)@WX
    y_kernel=np.transpose(Wy)@WX
    

    #let us combine L2 norm of the features lambda2*w^Tw and w^TX^TXw into a single term
    #lambda2 may be a vector of l2 weights or a scalar
    if (type(lambda2)==np.matrix)|(type(lambda2)==np.array):
        if len(lambda2.shape)==1:
            X_kernel = X_kernel + np.diag(lambda2)
        else:
            X_kernel = X_kernel + np.diagflat(lambda2)
    else: #if numeric
        X_kernel = X_kernel+lambda2*np.diag(np.ones(n_features))

    
    return X_kernel,y_kernel
 
#given the value of x, compute the value of the function, gradient, and the norm of the gradient   
def compute_values(X_kernel,y_kernel,w,lambda1=0):
    """
    Computes values of the function and its gradient
    Input: X_kernel = kernel matrix, np.matrix of size d-by-d
            y_kernel = kernel vector, np.matrix of size 1-by-d
            w = feature vector, np.matrix of size d-by-1
            lambda1 = L1 weights, np.matrix of size d-by-1 with nonnegative entries
    Output: func = value of the function, numeric
            grad = value of the subgradient, np.matrix of size 1-by-d        
    """
    #We compute that part of the function that depends on features w, i.e.,
    #w^T*X_kernel*w
    func = (np.transpose(w)@X_kernel@w)[0,0]-2*(y_kernel@w)[0,0]+lambda1*np.linalg.norm(w,ord=1)
    grad = 2*(np.transpose(w)@X_kernel)-2*y_kernel+lambda1*np.transpose(np.sign(w))
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
        
            

def grad_descent_constant_stepsize(
    X_kernel,y_kernel, w, lambda1=1, stepsize=-1, maxIter=1000000,precision=1e-5):
    """
    Name: grad_descent_constant_stepsize
    Goal: compute the minimum of the function
        w^T X_kernel w - 2 y_kernel w +lambda1||w||_1
        with respect to w. The minumum is found using the subgradient descent method
        with constant stepsize.
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial input value
        lambda1 = numeric, coefficient for L1 norm
        stepsize = numeric, value of the stepsize. Default =-1, i.e.,
            will be computed based on current value of the subgradient
        maxIter = integer, maximum number of iterations allowed, default =1000000
        precision = numeric, if least possible subgradient norm is less, we stop,default=1e-10
    Output:
        w = d-by-1 np.matrix, features when the value of the function is the least, first entry is bias
        func = numeric, least possible value of function
        converged = boolean, indicates if the method has converged
    """
    #initiate the gradient descent with current value of function, its derivative and norm    
    func, grad = compute_values(X_kernel, y_kernel, w, lambda1)
    grad_norm = grad_norm_min(grad,w,lambda1) #compute least potential norm of gradient
    iteration = 0
    change_w = 10 
    if stepsize<0:
        stepsize = 0.01/(np.linalg.norm(np.transpose(w)@X_kernel-2*y_kernel+lambda1*np.transpose(np.sign(w))))
        #We assume that the subgradient can grow at most 100 times larger than the
        #current value of the subgradient
    print("-----------------------------------------------------")
    print("Gradient descent with constant stepsize is initiated with the following parameters:")
    print("-----------------------------------------------------")
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration, grad_norm, func, np.linalg.norm(w)))
    while ((grad_norm>precision)&(iteration<maxIter)&(change_w>1e-5)):
        iteration+=1
        change_w = stepsize*grad_norm
        w = w-stepsize*np.transpose(grad)
        func, grad = compute_values(X_kernel,y_kernel,w,lambda1)
        grad_norm = grad_norm_min(grad,w,lambda1)

        if iteration in range(0,maxIter,50000):
            print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,np.linalg.norm(grad),func,np.linalg.norm(w)))
    #print parameters of the final iteration
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    
    if ((grad_norm<precision)|(change_w==0))&(func==func):
        print("The gradient descent converged after %s iterations." %iteration)
        converged = True
    elif (change_w<=1e-5)&(func==func):
        print("The gradient descent converged numerically, current change in w is too small (<1e-5)")
        converged = True
    else:
        print("The gradient descent did not converge.")
        converged = False
    
    print("-----------------------------------------------------")
    return w,func,grad_norm,converged
    

#function gradient descent
#with non-summable but square-summable sequence
#However, it was so slow that I decided to go with mixture model
def grad_descent_sequence(
    X_kernel,y_kernel,w,lambda1=1, stepsize=-1, maxIter=100000,precision=1e-5):
    """
    Name: grad_descent_sequence
    Goal: compute the minimum of the function
        w^T X_kernel w - 2 y_kernel w +lambda1||w||_1
        with respect to w. The minumum is found using the subgradient descent method
        with stepsize defined by sequence C/n.
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial input value
        lambda1 = numeric, coefficient for L1 norm
        stepsize = numeric, value of the stepsize. Default =-1, i.e.,
            will be computed based on the current value of the subgradient
        maxIter = integer, maximum number of iterations allowed, default =1000000
        precision = numeric, if least possible subgradient norm is less, we stop,default=1e-5
    Output:
        w = d-by-1 np.matrix, features when the value of the function is the least, first entry is bias
        func = numeric, least possible value of function
        converged = boolean, indicates if the method has converged
    """
    #initiate the gradient descent with current value of function, its derivative and norm    
    func, grad = compute_values(X_kernel, y_kernel, w, lambda1)
    grad_norm = grad_norm_min(grad,w,lambda1) #compute least potential norm of gradient
    iteration=0 
    change_w=10
    if stepsize<0:
        stepsize = 0.5/(np.linalg.norm(np.transpose(w)@X_kernel-2*y_kernel+lambda1*np.transpose(np.sign(w))))
    print("-----------------------------------------------------")
    print("The gradient descent using sequence stepsize/n is initiated")
    print("-----------------------------------------------------")
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))   
    while ((grad_norm>precision)&(iteration<maxIter)&(change_w>1e-5)):
        iteration+=1
        change_w = (stepsize/iteration)*grad_norm
        w = w-(stepsize/iteration)*np.transpose(grad)
        func,grad = compute_values(X_kernel,y_kernel,w,lambda1)
        grad_norm = grad_norm_min(grad,w,lambda1) #compute least potential norm of gradient           
        if iteration in range(0,maxIter,30000):
            print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    if ((grad_norm<precision)|(change_w==0))&(func==func):
        print("The gradient descent converged after %s iterations." %iteration)
        converged = True
    elif (change_w<=1e-5)&(func==func):
        print("The gradient descent converged numerically, current change in w is too small (<1e-5)")
        converged = True
    else:
        print("The gradient descent did not converge.")
        converged = False
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration, grad_norm, func, np.linalg.norm(w)))
    print("-----------------------------------------------------")
    return w, func, grad_norm, converged
    

#function gradient descent
#using Armijo search
def grad_descent_Armijo(X_kernel,y_kernel, w,lambda1=1, stepsize=-1,gamma=0.7, maxIter=500000,precision=1e-10,maxPower=10000):
    """
    Name: grad_descent_Armijo
    Goal: compute the minimum of the function
        w^T X_kernel w - 2 y_kernel w +lambda1||w||_1
        with respect to w. The minumum is found using the subgradient descent method
        with stepsize defined by Armijo search.
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial input value
        lambda1 = numeric, coefficient for L1 norm
        stepsize = numeric, value of the stepsize. Default =-1, i.e.,
            will be computed based on the current value of the subgradient
        gamma = numeric, value of gamma, default=0.7
        maxIter = integer, maximum number of iterations allowed, default =1000000
        precision = numeric, if least possible subgradient norm is less, we stop,default=1e-5
    Output:
        w = d-by-1 np.matrix, features when the value of the function is the least, first entry is bias
        func = numeric, least possible value of function
        converged = boolean, indicates if the method has converged
    """
    
    #initiate the gradient descent with current value of function, its derivative and norm    

    func, grad = compute_values(X_kernel,y_kernel,w,lambda1)
    grad_norm = grad_norm_min(grad,w,lambda1) #compute least potential norm of gradient
    iteration=0 
    change_w=10
    if stepsize<0:
        stepsize = 10/(np.linalg.norm(np.transpose(w)@X_kernel-2*y_kernel+lambda1*np.transpose(np.sign(w))))
    print("-----------------------------------------------------")
    print("The gradient descent using Armijo search is initiated")
    print("-----------------------------------------------------")
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))    
    while ((grad_norm>precision)&(iteration<maxIter)&(change_w>1e-5)):
        iteration+=1
        #Initiate Armijo search
        power=1
        gamma_power=1
        func_new=func
        while((power<=maxPower)&(func_new>=func)):
            change_w = stepsize*gamma_power*grad_norm
            w_new=w-stepsize*gamma_power*np.transpose(grad)
            func_new,grad_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
            grad_norm_new = grad_norm_min(grad_new,w_new,lambda1) #compute least potential norm of gradient
            gamma_power = gamma*gamma_power
            power+=1
        change_w = np.linalg.norm(w-w_new)
        w = w_new.copy()
        func = func_new.copy()
        grad = grad_new.copy()
        grad_norm = grad_norm_new.copy()    
        #if (power==maxPower+1):
        #    print("The Armijo search was not successful. Please relax the conditions")
        
        if iteration in range(0,maxIter,20000):
             print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    if (grad_norm<precision)|(change_w==0):
        print("The gradient descent converged after %s iterations." %iteration)
        converged = True
    elif (change_w<=1e-5):
        print("The gradient descent converged numerically, current change in w is too small (<1e-5)")
        converged = True
    else:
        print("The gradient descent did not converge.")
        converged = False
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    print("-----------------------------------------------------")
    return w,func,grad_norm,converged

#function gradient descent
#using line search
def grad_descent_line_search(X_kernel,y_kernel, w,lambda1=1,maxIter=100000,precision=1e-10,maxPower=10000):
    """
    Name: grad_descent_line_search
    Goal: compute the minimum of the function
        w^T X_kernel w - 2 y_kernel w +lambda1||w||_1
        with respect to w. The minumum is found using the subgradient descent method
        with stepsize defined by line search.
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial input value
        lambda1 = numeric, coefficient for L1 norm
        stepsize = numeric, value of the stepsize. Default =-1, i.e.,
            will be computed based on the current value of the subgradient
        gamma = numeric, value of gamma, default=0.7
        maxIter = integer, maximum number of iterations allowed, default =1000000
        precision = numeric, if least possible subgradient norm is less, we stop,default=1e-5
    Output:
        w = d-by-1 np.matrix, features when the value of the function is the least, first entry is bias
        func = numeric, least possible value of function
        converged = boolean, indicates if the method has converged
    """
    #initiate the gradient descent with current value of function, its derivative and norm    

    func,grad = compute_values(X_kernel,y_kernel,w,lambda1)
    grad_norm = grad_norm_min(grad,w,lambda1) #compute least potential norm of gradient
    iteration = 0
    change_w = 10  
    print("-----------------------------------------------------")
    print("The gradient descent using (analytic) line search is initiated")
    print("-----------------------------------------------------")  
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    while ((grad_norm>precision)&(iteration<maxIter)&(change_w>0)):
        iteration+=1
        #if the previous iteration resulted in too small change of w,
        #perhaps, we are at the point when the gradient does not exist,
        #and the subgradient gives vague result. To fix it, we jitter the point
        f_next = func.copy() #stores potential least value of function
        #In the line search, we seek for the stepsize that minimizes the derivative
        #along line f(w)+alpha*grad(f)
        #with respect to alpha
        #For the given problem, we can compute alpha analytically
        #(to get rid of extra negative signs when we compute the derivative,
        #we consider f(w)+alpha*grad(f) instead of f(w)-alpha*grad(f))
        
        #We need to minimize
        #(w+alpha*grad(f))^TX_kernel(w+alpha*grad(f))-2y_kernel(w+alpha*grad(f))+lambda1 ||w+alpha*grad(f)||_1
        #Or, if terms that do not include alpha are ommitted
        #alpha^2 grad(f)^T X_kernel grad(f)+2alpha w^T X_kernel grad(f)-2alpha*y_kernel grad(f)+lambda1||w+alpha*grad(f)||_1
        #=alpha^2 grad(f)^T X_kernel grad(f) +2alpha(w^T X_kernel-y_kernel)grad(f)+lambda1||w+alpha*grad(f)||_1
        
        #First of all, we need to check where the gradient is discontinous or d.n.e.
        #for the given function, it is when at least one of the entries of w+alpha*grad(f) is 0
        n_features = w.shape[0]
        #Going over all coordinates, we compute the function at each such point and find the smallest
        alphas=[]
        for i in range(n_features):
            if grad[0,i]!=0:
                potential_alpha = -w[i,0]/grad[0,i]
                alphas.append(potential_alpha)
                potential_w = w + potential_alpha*np.transpose(grad)
                potential_func,potential_grad = compute_values(X_kernel,y_kernel,potential_w,lambda1)
                potential_grad_norm = grad_norm_min(potential_grad,potential_w,lambda1) #compute least potential norm of gradient
                if potential_func<=f_next:
                    #Update the info about what value of w produces the least value of the function
                    w_next = potential_w.copy()
                    f_next = potential_func.copy()
                    gradient_next = potential_grad.copy()
                    grad_norm_next = potential_grad_norm.copy()
         
        #Second, we need to check when the derivative of
        # alpha^2 grad(f)^T X_kernel grad(f) +2alpha(w^T X_kernel-y_kernel)grad(f)+lambda1||w+alpha*grad(f)||_1
        #is 0
        #
        #The derivative is
        #2alpha grad(f)^T X_kernel grad(f)+2(w^T X_kernel-y_kernel)grad(f)+lambda1*grad(f) sign(w+alpha*grad(f))
        #Let coef1 = alpha grad(f)^T X_kernel grad(f)
        coef1 = ((grad@X_kernel)@np.transpose(grad))[0,0]
        if coef1<1e-20:
            print("Very badly conditioned matrix")
        #and coef2 = (w^T X_kernel-y_kernel)grad(f)
        coef2 = ((np.transpose(w)@X_kernel-y_kernel)@np.transpose(grad))[0,0]
        #Analytical solutions leads to
        # coef1*alpha=-coef2-0.5*lambda1*grad(f) sign(w+alpha*grad(f))
        coef2_scaled = coef2/coef1
        lambda1_scaled = 0.5*lambda1/coef1
        #Now we need to solve
        #alpha=-coef2_scaled -lambda1_scaled *grad(f) sign(w+alpha*grad(f))
        
        #To expand  $sign(w+alpha*grad(f))$, we need to know the sign of each entry
        #Note that it is a piece-wise constant function
        #However, each entry changes the sign at points when alpha=-w/grad(f), i.e.,
        #at the points stored at the variable alphas
        alphas = np.array(alphas)
        alphas=np.sort(alphas)
        #Now we need to go through each interval
        for i in range(len(alphas)):
            # consider alpha that is between alphas[i] and alphas[i+1]
            # if i+1>len(stepsize), assume that the value of alphas[i+1] is infinity
            
            # if alpha in (alphas[i],alphas[i+1]),
            # the value of $sign(w-alpha*subgradient)$ is unchanged
            # and equals the value at (stepsize[i]+stepsize[i+1])/2
            if i==(len(alphas)-1):
                midpoint = alphas[i]+1
            else:
                midpoint = (alphas[i] + alphas[i+1])/2
            signs = np.sign(w + midpoint*np.transpose(grad))
            potential_alpha = - coef2_scaled - lambda1_scaled*(grad@signs)[0,0]
            if (i<len(alphas)-1):
                if ((potential_alpha>alphas[i])&(potential_alpha<alphas[i+1])):
                    #print(potential_alpha)
                    potential_w = w + potential_alpha*np.transpose(grad)
                    potential_func,potential_grad = compute_values(X_kernel,y_kernel,potential_w,lambda1)
                    potential_grad_norm = grad_norm_min(potential_grad,potential_w,lambda1) #compute least potential norm of gradient
                    if potential_func<f_next:
                        #Update the info about what value of w produces the least value of the function
                        w_next = potential_w.copy()
                        f_next = potential_func.copy()
                        gradient_next = potential_grad.copy()
                        grad_norm_next = potential_grad_norm.copy()
            elif (potential_alpha>alphas[i]):
                #print(potential_stepsize)
                potential_w = w + potential_alpha*np.transpose(grad)
                potential_func,potential_grad = compute_values(X_kernel,y_kernel,potential_w,lambda1)
                potential_grad_norm = grad_norm_min(potential_grad,potential_w,lambda1) #compute least potential norm of gradient
                if potential_func<f_next:
                    #Update the info about what value of w produces the least value of the function
                    w_next = potential_w.copy()
                    f_next = potential_func.copy()
                    gradient_next = potential_grad.copy()
                    grad_norm_next = potential_grad_norm.copy()
                
    
#            
        #And finally check for values of alpha less than alphas[0]
        midpoint = alphas[0]-1
        signs = np.sign(w + midpoint*np.transpose(grad))
        potential_alpha = - coef2_scaled -lambda1_scaled*(grad@signs)[0,0]
        #print(potential_stepsize)
        potential_w = w + potential_alpha*np.transpose(grad)
        potential_func,potential_grad = compute_values(X_kernel,y_kernel,potential_w,lambda1)
        potential_grad_norm = grad_norm_min(potential_grad,potential_w,lambda1) #compute least potential norm of gradient
        if potential_func<f_next:
            #Update the info about what value of w produces the least value of the function
            w_next = potential_w.copy()
            f_next = potential_func.copy()
            gradient_next = potential_grad.copy()
            grad_norm_next = potential_grad_norm.copy()
        
        #Compute the amount of value w changes
        change_w = np.linalg.norm(w_next-w)
        #Update current values of w, function, gradient, and gradient norm
        w = w_next.copy()
        func = f_next.copy()
        grad = gradient_next.copy()
        grad_norm = grad_norm_next.copy()

        if iteration in range(0,maxIter,10000):
            #print(w)
            print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    

    if grad_norm<precision:
        print("The gradient descent converged after %s iterations." %iteration)
        converged = True
    elif (change_w==0):
        print("The gradient descent converged numerically, current change in w is too small")
        converged = True
    else:
        print("The gradient descent did not converge, insufficient number of iterations")
        converged = False
    print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    print("-----------------------------------------------------")
    return w,func,grad_norm,converged

    
def _test(n_features=8,n_samples=100,warm_start=True):
    indexes=np.random.randint(0,n_features,3)
    true_w =np.zeros((n_features+1,1))
    true_w[indexes,:]=np.matrix(np.random.randn(3,1))
    true_w[0,:] = np.random.randn(1)
    X = np.random.randn(n_samples,n_features+1)
    X[:,0]=np.ones(n_samples)

    error=np.random.randn(n_samples,1)
    y=X@true_w+error
    #X = pd.DataFrame(X)

    lambda1=3
    lambda2=1
    X_kernel,y_kernel = compute_kernels(X,y,lambda2)
    
    if warm_start==False:
        w = np.random.randn(n_features+1,1)
    else:
        #warm start
        w = np.linalg.pinv(X_kernel)@np.transpose(y_kernel)
    print("Initial norm of error %s"%np.linalg.norm(true_w-w))
    
    stepsize = 0.01/(np.linalg.norm(np.transpose(w)@X_kernel-2*y_kernel+lambda1*np.transpose(np.sign(w))))
    print("Initial stepsize %s"%stepsize)
    
    #w_min,f_min,grad_norm,converged =grad_descent_constant_stepsize(X_kernel,y_kernel,w,lambda1=lambda1, stepsize=stepsize)
    #print("Norm of error %s"%np.linalg.norm(true_w-w_min))
    
    #stepsize = 0.5/(np.linalg.norm(np.transpose(w)@X_kernel-2*y_kernel+lambda1*np.transpose(np.sign(w))))
    #print("Initial stepsize %s"%stepsize)
    #w_min,f_min,grad_norm,converged =grad_descent_sequence(X_kernel,y_kernel,w,lambda1=lambda1,stepsize=stepsize)
    #print("Norm of error %s"%np.linalg.norm(true_w-w_min))
    
    w_min,f_min,grad_norm,converged =grad_descent_Armijo(X_kernel,y_kernel,w,lambda1=lambda1)
    print("Norm of error %s"%np.linalg.norm(true_w-w_min))
    
    x_min,f_min,grad_norm,converged = grad_descent_line_search(X_kernel,y_kernel,w,lambda1=lambda1)
    print("Norm of error %s"%np.linalg.norm(true_w-x_min))
    
    print("Initial norm of error %s"%np.linalg.norm(true_w-w))

    #x_min,f_min,grad_norm = grad_descent_BB_stepsize(X_kernel,y_kernel,x,lambda1=lambda1)
    #print("Norm of error %s"%np.linalg.norm(true_w-x_min))
    
    
    