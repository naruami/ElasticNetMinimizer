#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:32:20 2017

@author: Kateryna Melnykova


"""

import numpy as np
#To speed up computations, I combine approx. error and L2 norm into the single formula
def compute_kernels(X,y,lambda2,weights=None):
    n_features=X.shape[1]
    #define weighted samples and regression matrix
    if np.any(weights):
        weights=np.diagflat(weights)
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
    """
    Name: compute_values
    Goal: compute value of the function 
        w X_kernel - 2 y_kernel w +lambda1||w||_1
        at point w=w
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, input value
        lambda1 = numeric, coefficient for L1 norm
    Output:
        func = numeric, value of the function
    """
    func=(np.transpose(w)@X_kernel@w)[0,0]-2*(y_kernel@w)[0,0]+lambda1*np.linalg.norm(w,ord=1)
    return func

def function_update(X_kernel,y_kernel,w_old,no_feature,alpha,lambda1):
    #Currently, unused function -- will incorporate it soon
    """
    Name: compute_values
    Goal: compute the amount of change of the function 
        w X_kernel - 2 y_kernel w +lambda1||w||_1
        from w=w_old to 
        w=w_old on all features except no_feature, but w[no_feature]=w_old[no_feature]+alpha
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w_old = d-by-1 np.matrix, first input value
        no_feature = integer,number of feature updated
        alpha = numeric, amount of update of the feature
        lambda1 = numeric, coefficient for L1 norm
    Output:
        diff = numeric, difference in values
    """
    #Let w_new be a new value of the input vector. Note that
    #w_new = w_old +alpha e_i,
    #where i=no_feature and e_i is a basic vector which entries are 0 except
    #it's i-th entry is 1. Then
    #f(w_new) =
    #     (w_old+alpha e_i)^T X_kernel (w_old+alpha e_i)
    #      - 2y_kernel (w_old +alpha e_i)
    #      + lambda1 ||w_old+alpha e_i||_1
    #Denote the i-th entry of z by z_i, and (X_kernel)_i corresponds to the i-th
    #row of X_kernel.Simplifying the function and using that w_new and w_old
    #coincide for most of the entries, so 1-norm may be also simplified
    #f(w_new) =
    #       w_old^T X_kernel w_old +2alpha (X_kernel)_i w_old +alpha^2 (X_kernel)_{i,i}
    #       -2y_kernel w_old - 2alpha (y_kernel)_i
    #       +lambda1 ||w_old||_1 +lambda1|(w_old)_i+alpha|-lambda1 |(w_old)_i|
    #
    #       = f(w_old)
    #       + 2alpha (X_kernel)_i w_old +alpha^2 (X_kernel)_{i,i}
    #       - 2alpha (y_kernel)_i
    #       +lambda1|(w_old)_i+alpha|-lambda1 |(w_old)_i|
    #Since we are looking for f(w_new)-f(w_old), we need to compute
    # 2alpha (X_kernel)_i w_old +alpha^2 (X_kernel)_{i,i}
    #       - 2alpha (y_kernel)_i
    #       +lambda1|(w_old)_i+alpha|-lambda1 |(w_old)_i|
    #Note that now there is no matrix multiplication, only vector multiplication
    #which is computationally faster
    
    diff = 2*alpha*(X_kernel[no_feature,:]@w_old)[0,0] +alpha**2 *X_kernel[no_feature,no_feature]
    diff += -2*alpha*y_kernel[0,no_feature]
    diff += lambda1*np.abs(w_old[no_feature,0]+alpha)-lambda1*np.abs(w_old[no_feature,0])
    return diff
    
    
    
#given the value of x, compute the value of the function, gradient, and the norm of the gradient   
def compute_values_with_gradient(X_kernel,y_kernel,w,lambda1):
    """
    Name: compute_values_with_gradient
    Goal: compute values of the function, subgradient, and least possible subgradient norm
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, value of the input
        lambda1 = numeric, coefficient for L1 norm
    Output:
        func = value of the function
         w^T X_kernel w - 2 y_kernel w +lambda1 ||w||_1
        grad = subgradient of the function
        grad_norm = least potential norm of subgradient at the given point
    """
    #We compute that part of the function that depends on features w, i.e.,
    #w^T*X_kernel*w
    func=(np.transpose(w)@X_kernel@w)[0,0]-2*(y_kernel@w)[0,0]+lambda1*np.linalg.norm(w,ord=1)
    grad=2*(np.transpose(w)@X_kernel)-2*y_kernel+lambda1*np.transpose(np.sign(w))
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
    return func, grad, np.sqrt(grad_norm_squared)
    


def stochastic_gradient_descent_deterministic_order(
    X_kernel,y_kernel,w,lambda1=1, maxIter=1000,precision=1e-5):
    """
    Name: Stochastic_gradient_descent_deterministic_order
    Goal: Compute minimum of elastic net regularizer, i.e., to find the minimum
        of the function
        w^T X^T W^2 X w - 2y^T W^2 X w +lambda2||w||_2^2 +lambda1||w||_1 
        with respect to w.
        For each iteration, we compute the value of the subgradient, and
        we update one entry of w. The order of entries of w being updated is
        cycled deterministically
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial value of the predictor
        lambda1 = numeric, coefficient of L1 norm
        lambda2 = numeric, coefficient of L2 norm squared
        maxIter = integer, the largest number of iterations allowed, 1000 by Default
        precision = numeric, the value of the norm of the subgradient when we stop
    Output:
        w = d-by-1 np.matrix, the value of w which minimizes the function
        func = numeric,the least value of the function
        converged = boolean, if convergence is achieved
    """
    func = compute_values(X_kernel,y_kernel,w,lambda1)
    iteration = 0
 
    print("-----------------------------------------------------")
    print("Stochastic gradient descent with deterministic order of features is initiated.")
    print("-----------------------------------------------------")
    print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
    change_w=100
    while ((iteration<maxIter)&(change_w>0)):
        iteration+= 1
        change_w = 0
        for update_feature_no in range(0,w.shape[0]):
            # we seek to minimize the function 
            #f(w)=w^T X_kernel w -2*y_kernel w+lambda1*||w||_1
            #along the line
            #w+alpha*e_i, alpha is any real number
            
            #initial value of function f is f_current that corresponds to alpha=0
            w_next = w.copy()
            func_next=func.copy()
            
        
            #minimize
            #(w+alpha*e_i)^T X_kernel (w+alpha*e_i)-2y_kernel(w+alpha*e_i)+lambda1||w+alpha*e_i||
            
            #if we get rid of all terms that does not depend on alpha, we get
            #alpha^2 e_i^T X_kernel e_i +2alpha e_iX_kernel w-2alpha y_kernel*e_i+lambda_1|w_i+alpha|
            
            #This function is continuous everywhere and differentiable everywhere except alpha=-w_i 
            
            #We can compute the minimum analytically using differentiation
            #Firstly, the minimum may occur when derivative is discontinuous or d.n.e.
            #alpha =-w_i
            #Compute corresponding value of w
            w_new = w.copy()
            w_new[update_feature_no,0] = 0
            #Compute the value of function, its gradient and the norm if it is not identical
            #to the previous step
            if w[update_feature_no,0]!=0:
                func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
                if func_new<func_next:
                    w_next = w_new.copy()
                    func_next = func_new.copy()

            #Secondly, we need to compute zeros of the derivative.
            #Recall that we are minimizing
            #alpha^2 e_i^T X_kernel e_i +2alpha e_iX_kernel w-2alpha y_kernel*e_i+lambda_1|w_i+alpha|
            #The derivative (which exists for all alpha!=-w_i) is
            #2alpha e_i^T X_kernel e_i +2 e_i X_kernel w -2 y_kernel e_i+lambda1* sign(w_i+alpha)
            
            #Let coef1 = e_i^T X_kernel e_i which is (i,i)th entry of the matrix
            coef1 = X_kernel[update_feature_no,update_feature_no]
            #Let coef2 = -w^TX_kernel e_i + y_kernel*e_i
            coef2 = -(np.transpose(w)@(X_kernel[:,update_feature_no]))[0]+y_kernel[0,update_feature_no]
            if coef1<=0:
                print("Problem with X_kernel; it is not positive semi-definite matrix")
            #Now we need to solve
            #alpha*coef1 = coef2+0.5*lambda1 sign(w_i+alpha)
            #Let us divide everything on the coef1
            coef2_scaled = coef2/coef1
            lambda1_scaled = 0.5*lambda1/coef1
            #We have two cases alpha>-w_i and alpha<-w_i
            
            #Case 1: alpha>-w_i
            alpha = coef2_scaled + lambda1_scaled
            if (alpha+w[update_feature_no,0]>0)&(alpha!=0):
                #w_new = w.copy()
                w_new[update_feature_no,0] = w[update_feature_no,0]+ alpha
                func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
                if func_new<func_next:
                    w_next = w_new.copy()
                    func_next = func_new.copy()
#                   print("Case 1 update")
            
            #Case 2: alpha<-w_i
            alpha = coef2_scaled - lambda1_scaled
            if (alpha+w[update_feature_no,0]<0)&(alpha!=0):
                #w_new = w.copy()
                w_new[update_feature_no,0] = w[update_feature_no,0]+ alpha
                func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
                if func_new<func_next:
                    w_next = w_new.copy()
                    func_next = func_new.copy()
        
            #Update feature vector and value of function
            if np.linalg.norm(w-w_next)!=0:
                change_w +=(w[update_feature_no,0]-w_next[update_feature_no,0])**2
                w = w_next.copy()
                func = func_next.copy()

        #print("iter=%s||norm(w)=%s||f_cur=%s"%(iteration,np.linalg.norm(w),func))
        if iteration in range(0,maxIter,2000):
            print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
            #func1,grad1,grad_norm = compute_values_with_gradient(X_kernel,y_kernel,w,lambda1)
            #print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
    if (change_w==0):
        print("The stochastic gradient descent with deterministic order of features updated converged.")
        converged = True
    elif (iteration<=maxIter):
        print("The stochastic gradient descent numerically converged, current update <1e-10." )
        converged = True
    else:
        print("The stochastic gradient descent with deterministic order of features updated did not converge, insufficient number of iterations")
        converged = False
    print("-----------------------------------------------------")
    return w,func, converged

def check_if_approx_minimum(X_kernel,y_kernel,w,lambda1,precision=1e-10):
    minimum = True
    feature = -1
    alpha_stored = 0
    func = compute_values(X_kernel,y_kernel,w,lambda1)
    precision /=np.sqrt(w.shape[0])
    for update_feature_no in range(0,w.shape[0]):
        # we seek to minimize the function 
        #f(w)=w^T X_kernel w -2*y_kernel w+lambda1*||w||_1
        #along the line
        #w+alpha*e_i, alpha is any real number
                   
        
        #minimize
        #(w+alpha*e_i)^T X_kernel (w+alpha*e_i)-2y_kernel(w+alpha*e_i)+lambda1||w+alpha*e_i||
        
        #if we get rid of all terms that does not depend on alpha, we get
        #alpha^2 e_i^T X_kernel e_i +2alpha e_iX_kernel w-2alpha y_kernel*e_i+lambda_1|w_i+alpha|
        
        #This function is continuous everywhere and differentiable everywhere except alpha=-w_i 
            
        #We can compute the minimum analytically using differentiation
        #Firstly, the minimum may occur when derivative is discontinuous or d.n.e.
        #alpha =-w_i
        #Compute corresponding value of w
        w_new = w.copy()
        w_new[update_feature_no,0] = 0
        #Compute the value of function, its gradient and the norm if it is not identical
        #to the previous step
        if np.abs(w[update_feature_no,0])>precision:
            func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
            if (func_new<func):
                minimum = False
                feature = update_feature_no
                alpha_stored = -w[update_feature_no,0]
        #Secondly, we need to compute zeros of the derivative.
        #Recall that we are minimizing
        #alpha^2 e_i^T X_kernel e_i +2alpha e_iX_kernel w-2alpha y_kernel*e_i+lambda_1|w_i+alpha|
        #The derivative (which exists for all alpha!=-w_i) is
        #2alpha e_i^T X_kernel e_i +2 e_i X_kernel w -2 y_kernel e_i+lambda1* sign(w_i+alpha)
            
        #Let coef1 = e_i^T X_kernel e_i which is (i,i)th entry of the matrix
        coef1 = X_kernel[update_feature_no,update_feature_no]
        #Let coef2 = -w^TX_kernel e_i + y_kernel*e_i
        coef2 = -(np.transpose(w)@(X_kernel[:,update_feature_no]))[0]+y_kernel[0,update_feature_no]
        if coef1<=0:
            print("Problem with X_kernel; it is not positive semi-definite matrix")
        #Now we need to solve
        #alpha*coef1 = coef2+0.5*lambda1 sign(w_i+alpha)
        #Let us divide everything on the coef1
        coef2_scaled = coef2/coef1
        lambda1_scaled = 0.5*lambda1/coef1
        #We have two cases alpha>-w_i and alpha<-w_i
            
        #Case 1: alpha>-w_i
        alpha = coef2_scaled + lambda1_scaled
        if (alpha+w[update_feature_no,0]>0)&(np.abs(alpha)>precision):
            #w_new = w.copy()
            w_new[update_feature_no,0] = w[update_feature_no,0]+ alpha
            func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
            if func_new<func:
                minimum = False
                feature = update_feature_no
                alpha_stored = alpha 
            
        #Case 2: alpha<-w_i
        alpha = coef2_scaled - lambda1_scaled
        if (alpha+w[update_feature_no,0]<0)&(np.abs(alpha)>precision):
            #w_new = w.copy()
            w_new[update_feature_no,0] = w[update_feature_no,0]+ alpha
            func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
            if func_new<func:
                minimum = False
                feature = update_feature_no
                alpha_stored = alpha 
        
    return minimum,feature, alpha_stored
    
def stochastic_gradient_descent_uni_random_order(
    X_kernel,y_kernel,w,lambda1=1, maxIter=100000,precision=1e-10):
    """
    Name: Stochastic_gradient_descent_uni_random_order
    Goal: Compute minimum of elastic net regularizer, i.e., to find the minimum
        of the function
        w^T X^T W^2 X w - 2y^T W^2 X w +lambda2||w||_2^2 +lambda1||w||_1 
        with respect to w.
        For each iteration, we update a random entry of the predictor to make
        the function as least as possible
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial value of the predictor
        lambda1 = numeric, coefficient of L1 norm
        lambda2 = numeric, coefficient of L2 norm squared
        maxIter = integer, the largest number of iterations allowed, 1000 by Default
        precision = numeric, the value of the norm of the subgradient when we stop
    Output:
        w = d-by-1 np.matrix, the value of w which minimizes the function
        func = numeric,the least value of the function
        converged = boolean, if convergence is achieved
    """
    func=compute_values(X_kernel,y_kernel,w,lambda1)
    iteration=0
 
    print("-----------------------------------------------------")
    print("Stochastic gradient descent with features chosen uniformly is initiated.")
    print("-----------------------------------------------------")
    print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
    n_features = w.shape[0]
    no_change = 0 #How many iterations there was no update to w
    minimum = False
    while ((iteration < maxIter)&(minimum == False)):
        iteration+= 1
        no_change+= 1
        #decide which feature to update
        update_feature_no = np.random.randint(0,n_features-1)
            
        # we seek to minimize the function 
        #f(w)=w^T X_kernel w -2*y_kernel w+lambda1*||w||_1
        #along the line
        #w+alpha*e_i, alpha is any real number
            
        #initial value of function f is f_current that corresponds to alpha=0
        w_next = w.copy()
        func_next=func.copy()
            
        
        #minimize
        #(w+alpha*e_i)^T X_kernel (w+alpha*e_i)-2y_kernel(w+alpha*e_i)+lambda1||w+alpha*e_i||
            
        #if we get rid of all terms that does not depend on alpha, we get
        #alpha^2 e_i^T X_kernel e_i +2alpha e_iX_kernel w-2alpha y_kernel*e_i+lambda_1|w_i+alpha|
            
        #This function is continuous everywhere and differentiable everywhere except alpha=-w_i 
            
        #We can compute the minimum analytically using differentiation
        #Firstly, the minimum may occur when derivative is discontinuous or d.n.e.
        #alpha =-w_i
        #Compute corresponding value of w
        w_new = w.copy()
        w_new[update_feature_no,0] = 0
        #Compute the value of function, its gradient and the norm if it is not identical
        #to the previous step
        if w[update_feature_no,0]!=0:
            func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
            if func_new<func_next:
                no_change = 0
                w_next = w_new.copy()
                func_next = func_new.copy()

        #Secondly, we need to compute zeros of the derivative.
        #Recall that we are minimizing
        #alpha^2 e_i^T X_kernel e_i +2alpha e_iX_kernel w-2alpha y_kernel*e_i+lambda_1|w_i+alpha|
        #The derivative (which exists for all alpha!=-w_i) is
        #2alpha e_i^T X_kernel e_i +2 e_i X_kernel w -2 y_kernel e_i+lambda1* sign(w_i+alpha)
            
        #Let coef1 = e_i^T X_kernel e_i which is (i,i)th entry of the matrix
        coef1 = X_kernel[update_feature_no,update_feature_no]
        #Let coef2 = -w^TX_kernel e_i + y_kernel*e_i
        coef2 = -(np.transpose(w)@(X_kernel[:,update_feature_no]))[0]+y_kernel[0,update_feature_no]
        if coef1<=0:
            print("Problem with X_kernel; it is not positive semi-definite matrix")
        #Now we need to solve
        #alpha*coef1 = coef2+0.5*lambda1 sign(w_i+alpha)
        #Let us divide everything on the coef1
        coef2_scaled = coef2/coef1
        lambda1_scaled = 0.5*lambda1/coef1
        #We have two cases alpha>-w_i and alpha<-w_i
            
        #Case 1: alpha>-w_i
        alpha = coef2_scaled + lambda1_scaled
        if (alpha+w[update_feature_no,0]>0)&(alpha!=0):
            #w_new = w.copy()
            w_new[update_feature_no,0] = w[update_feature_no,0]+ alpha
            func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
            if func_new<func_next:
                no_change = 0
                w_next = w_new.copy()
                func_next = func_new.copy()

#                print("Case 1 update")
            
        #Case 2: alpha<-w_i
        alpha = coef2_scaled - lambda1_scaled
        if (alpha+w[update_feature_no,0]<0)&(alpha!=0):
            #w_new = w.copy()
            w_new[update_feature_no,0] = w[update_feature_no,0]+ alpha
            func_new = compute_values(X_kernel,y_kernel,w_new,lambda1)
            if func_new<func_next:
                no_change = 0
                w_next = w_new.copy()
                func_next = func_new.copy()


        
        #Update feature vector and value of function
        if w[update_feature_no,0]!=w_next[update_feature_no,0]:
            w = w_next.copy()
            func = func_next.copy()
        #print("iter=%s||norm(w)=%s||f_cur=%s||feature_no=%s"%(iteration,np.linalg.norm(w),func,update_feature_no))
        if iteration in range(0,maxIter,20000):
            print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
            #Next two lines allow to check the gradient. Note: slows down the algorithm
            #func1,grad1,grad_norm = compute_values_with_gradient(X_kernel,y_kernel,w,lambda1)
            #print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    
        if no_change>0.3*n_features:
            #print("Suspected minumum at iteration %s"%iteration)
            minimum, feature_no, alpha = check_if_approx_minimum(X_kernel,y_kernel,w,lambda1)
            if minimum == False:
                w[feature_no,0]+= alpha
                func = compute_values(X_kernel,y_kernel,w,lambda1)
                no_change = 0 #the update of feature vector was just performed
    
    
    print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
    if (minimum==True):
        print("The stochastic gradient descent with features chosen uniformly numerically converged, current update <1e-10.")
        converged = True
        #print(w)
    else:
        print("The stochastic gradient descent with features chosen uniformly did not converge, insufficient number of iterations")
        converged = False
    print("-----------------------------------------------------")
    return w,func,converged

def stochastic_gradient_descent_gradient_order(
    X_kernel,y_kernel,w,lambda1=1,maxIter=1000,precision=1e-5):
    """
    Name: Stochastic_gradient_descent_gradient order
    Goal: Compute minimum of elastic net regularizer, i.e., to find the minimum
        of the function
        w^T X^T W^2 X w - 2y^T W^2 X w +lambda2||w||_2^2 +lambda1||w||_1 
        with respect to w.
        For each iteration, we compute the value of the subgradient, and
        we update the entry of w which corresponds to the largest entry.
    Input:
        X_kernel = d-by-d np.matrix, kernel matrix
        y_kernel = 1-by-d np.matrix, kernel vector
        w = d-by-1 np.matrix, initial value of the predictor
        lambda1 = numeric, coefficient of L1 norm
        lambda2 = numeric, coefficient of L2 norm squared
        maxIter = integer, the largest number of iterations allowed, 1000 by Default
        precision = numeric, the value of the norm of the subgradient when we stop
    Output:
        w = d-by-1 np.matrix, the value of w which minimizes the function
        func = numeric,the least value of the function
        converged = boolean, if convergence is achieved
    """
    func, grad, grad_norm = compute_values_with_gradient(X_kernel,y_kernel,w,lambda1)
    iteration=0
 
    print("-----------------------------------------------------")
    print("Stochastic gradient descent with features chosen according to the gradient values is initiated.")
    print("-----------------------------------------------------")
    print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
    n_features = w.shape[0]
    no_change = 0 #How many iterations there was no update to w
    minimum = False
    while ((iteration < maxIter)&(minimum == False)):
        iteration+= 1
        no_change+= 1
        #decide which feature to update
        update_feature_no = np.argmax(np.abs(grad))
            
        # we seek to minimize the function 
        #f(w)=w^T X_kernel w -2*y_kernel w+lambda1*||w||_1
        #along the line
        #w+alpha*e_i, alpha is any real number
            
        #initial value of function f is f_current that corresponds to alpha=0
        w_next = w.copy()
        func_next=func.copy()
            
        
        #minimize
        #(w+alpha*e_i)^T X_kernel (w+alpha*e_i)-2y_kernel(w+alpha*e_i)+lambda1||w+alpha*e_i||
            
        #if we get rid of all terms that does not depend on alpha, we get
        #alpha^2 e_i^T X_kernel e_i +2alpha e_iX_kernel w-2alpha y_kernel*e_i+lambda_1|w_i+alpha|
            
        #This function is continuous everywhere and differentiable everywhere except alpha=-w_i 
            
        #We can compute the minimum analytically using differentiation
        #Firstly, the minimum may occur when derivative is discontinuous or d.n.e.
        #alpha =-w_i
        #Compute corresponding value of w
        w_new = w.copy()
        w_new[update_feature_no,0] = 0
        #Compute the value of function, its gradient and the norm if it is not identical
        #to the previous step
        if w[update_feature_no,0]!=0:
            func_new,grad_new,grad_norm_new = compute_values_with_gradient(X_kernel,y_kernel,w_new,lambda1)
            if func_new<func_next:
                no_change = 0
                w_next = w_new.copy()
                func_next = func_new.copy()
                grad_next = grad_new.copy()
        #Secondly, we need to compute zeros of the derivative.
        #Recall that we are minimizing
        #alpha^2 e_i^T X_kernel e_i +2alpha e_iX_kernel w-2alpha y_kernel*e_i+lambda_1|w_i+alpha|
        #The derivative (which exists for all alpha!=-w_i) is
        #2alpha e_i^T X_kernel e_i +2 e_i X_kernel w -2 y_kernel e_i+lambda1* sign(w_i+alpha)
            
        #Let coef1 = e_i^T X_kernel e_i which is (i,i)th entry of the matrix
        coef1 = X_kernel[update_feature_no,update_feature_no]
        #Let coef2 = -w^TX_kernel e_i + y_kernel*e_i
        coef2 = -(np.transpose(w)@(X_kernel[:,update_feature_no]))[0]+y_kernel[0,update_feature_no]
        if coef1<=0:
            print("Problem with X_kernel; it is not positive semi-definite matrix")
        #Now we need to solve
        #alpha*coef1 = coef2+0.5*lambda1 sign(w_i+alpha)
        #Let us divide everything on the coef1
        coef2_scaled = coef2/coef1
        lambda1_scaled = 0.5*lambda1/coef1
        #We have two cases alpha>-w_i and alpha<-w_i
            
        #Case 1: alpha>-w_i
        alpha = coef2_scaled + lambda1_scaled
        if (alpha+w[update_feature_no,0]>0)&(alpha!=0):
            #w_new = w.copy()
            w_new[update_feature_no,0] = w[update_feature_no,0]+ alpha
            func_new,grad_new,grad_norm_new = compute_values_with_gradient(X_kernel,y_kernel,w_new,lambda1)
            if func_new<func_next:
                no_change = 0
                w_next = w_new.copy()
                func_next = func_new.copy()
                grad_next = grad_new.copy()
                
                

#                print("Case 1 update")
            
        #Case 2: alpha<-w_i
        alpha = coef2_scaled - lambda1_scaled
        if (alpha+w[update_feature_no,0]<0)&(alpha!=0):
            #w_new = w.copy()
            w_new[update_feature_no,0] = w[update_feature_no,0]+ alpha
            func_new,grad_new,grad_norm_new = compute_values_with_gradient(X_kernel,y_kernel,w_new,lambda1)
            if func_new<func_next:
                no_change = 0
                w_next = w_new.copy()
                func_next = func_new.copy()
                grad_next = grad_new.copy()

        
        #Update feature vector and value of function
        if w[update_feature_no,0]!=w_next[update_feature_no,0]:
            w = w_next.copy()
            func = func_next.copy()
            grad = grad_next.copy()
        #print("iter=%s||norm(w)=%s||f_cur=%s||feature_no=%s"%(iteration,np.linalg.norm(w),func,update_feature_no))
        if iteration in range(0,maxIter,20000):
            print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
            #Next two lines allow to check the gradient. Note: slows down the algorithm
            #func1,grad1,grad_norm = compute_values_with_gradient(X_kernel,y_kernel,w,lambda1)
            #print("Iteration=%s|gradient norm=%f|func value=%f||norm(w)=%s"%(iteration,grad_norm,func,np.linalg.norm(w)))
    
        if no_change>0.3*n_features:
            #print("Suspected minumum at iteration %s"%iteration)
            minimum, feature_no, alpha = check_if_approx_minimum(X_kernel,y_kernel,w,lambda1)
            if minimum == False:
                w[feature_no]+= alpha
                func,grad,grad_norm_new = compute_values_with_gradient(X_kernel,y_kernel,w,lambda1)
                no_change = 0 #the update of feature vector was just performed
    
    
    print("Iteration=%s|func value=%f||norm(w)=%s"%(iteration,func,np.linalg.norm(w)))
    if (minimum==True):
        print("The stochastic gradient descent converged, current update <1e-10.")
        converged = True
        #print(w)
    else:
        print("The stochastic gradient descent did not converge, insufficient number of iterations")
        converged = False
    print("-----------------------------------------------------")
    return w,func,converged

#def stochastic_gradient_descent(X,y,lambda1,lambda2,w,method='deterministic', bias='True'...
#                                maxIter=100000):
#    """
#    This function allows to compute the minimum of the elastic net function using
#    stochastic gradient descent.
#    Input:
#        X = N-by-d np.matrix, the regression matrix
#        y = N-by-1 np.matrix, the output variable
#        lambda1 = coefficient for L1 norm of predictors, numeric
#        lambda2 = coefficient for L2 norm of predictors, numeric
#        w = initial valueof the predictors, including bias
#        method = what method is used to compute the minimum, choose one of
#                'deterministic','uniform', 'gradient', default is 'deterministic'
#        bias = 
#    
#    """
def _test(n_features=10,n_samples=101,warm_start=False):
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
        x = np.random.randn(n_features+1,1)
    else:
        #warm start
        x = np.linalg.pinv(X_kernel)@np.transpose(y_kernel)
    print("Initial norm of error %s"%np.linalg.norm(true_w-x))
    
    #x_min,f_min,converged = stochastic_gradient_descent_deterministic_order(X_kernel,y_kernel,x,lambda1=lambda1)
    #print("Norm of error %s"%np.linalg.norm(true_w-x_min))
    
    x_min,f_min,converged = stochastic_gradient_descent_uni_random_order(X_kernel,y_kernel,x,lambda1=lambda1)
    print("Norm of error %s"%np.linalg.norm(true_w-x_min))
    
    x_min,f_min,converged = stochastic_gradient_descent_gradient_order(X_kernel,y_kernel,x,lambda1=lambda1)
    print("Norm of error %s"%np.linalg.norm(true_w-x_min))
    
    print("Initial norm of error %s"%np.linalg.norm(true_w-x))




    
