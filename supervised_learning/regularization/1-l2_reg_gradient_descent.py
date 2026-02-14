#!/usr/bin/env python3
import numpy as np

def l2_reg_gradient_descent(Y, P, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using 
    gradient descent with L2 regularization.
    """
    m = Y.shape[1]
    dZ = P - Y
    
    for i in range(L, 0, -1):
        A_prev = cache[f'A{i-1}']
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        
        # Calculate gradients
        dW = (1 / m) * np.matmul(dZ, A_L_minus_1.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        # Calculate dZ for next iteration (backpropagate)
        if i > 1:
            A_prev = cache[f'A{i-1}']
            dZ = np.matmul(W.T, dZ) * (1 - A_prev**2) # assuming tanh
            
        # Update weights and biases
        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db
        
    return weights
