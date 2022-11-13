from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W) #(D,C)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    dim = X.shape[1]
    cls = W.shape[1]

    score = np.zeros((num_train,cls)) #  X @ W #(N, C)
    for n in range(num_train):
      for c in range(cls):
        for d in range(dim):
          score[n,c] += X[n,d]*W[d,c]
    score -= np.max(score, axis = 1, keepdims = True)
    score_exp = np.exp(score)
    
    for i in np.arange(num_train):
      loss += -score[i,y[i]]
      dW[:,y[i]] -= X[i]

      loss += np.log(np.sum(score_exp[i]))


      expsum = np.sum(score_exp[i])
      for c in range(cls):
        dW[:,c] += X[i] * np.exp(score[i,c])/expsum
      

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2* reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    score = X @ W
    
    exp_score = np.exp(score) # (N,C)


    margin = np.zeros_like(score)
    margin = exp_score / (np.sum(exp_score, axis = 1).reshape(-1,1))
    margin[np.arange(N),y] -= 1
    # print(margin.shape)


    loss_vector = -score[np.arange(N),y] + np.log(np.sum(exp_score,axis = 1))
    loss = np.sum(loss_vector)/N
    loss += reg * np.sum(W * W)

    dW = X.T @ margin
    dW /= N
    dW += 2*reg*W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
