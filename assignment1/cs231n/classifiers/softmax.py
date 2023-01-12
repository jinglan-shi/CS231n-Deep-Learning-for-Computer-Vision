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
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_examples = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_examples):
      scores = X[i].dot(W)
      shifted_scores = scores - np.max(scores)
      scores_norm = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores))
      loss_i = -np.log(scores_norm[y[i]])
      loss += loss_i
      for j in range(num_class):
        if j == y[i]:
          dlf = -1 + scores_norm[j]
          dW[:,j] += X[i].T * dlf
        else:
          dlf = scores_norm[j]
          dW[:,j] += X[i].T * dlf

    dW = dW/num_examples + reg*W
    
    loss /= num_examples
    loss += 0.5*reg*np.sum(W*W)
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
    num_examples = X.shape[0]
    num_class = W.shape[1]

    scores = X.dot(W)
    shifted_scores = scores - np.max(scores)
    scores_norm = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis=1, keepdims=True)

    correct_probs = scores_norm[range(num_examples), y]
    loss = np.sum(-np.log(correct_probs))/num_examples + 0.5*reg*np.sum(W*W)

    dscores = scores_norm
    dscores[range(num_examples), y] -= 1 
    dW += X.T.dot(dscores)
    dW = dW/num_examples + reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
