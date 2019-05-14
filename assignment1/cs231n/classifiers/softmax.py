import numpy as np
from random import shuffle

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
  #pass
  dW_one_sample = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  dimension = X.shape[1]

  for i in range(num_train):
    f = X[i,:].dot(W)    # A numpy array of shape (1, C)    
    true_distribution = np.zeros_like(f)   # A numpy array of shape (1, C)
    true_distribution[y[i]] = 1
    logC = - np.max(f)
    P = np.exp(f + logC) / np.sum(np.exp(f + logC)) # A numpy array of shape (1, C)
    loss = loss + (-1 * np.sum(true_distribution * np.log(P)))

    for j in range(num_classes):
      if j == y[i]:
        dW_one_sample[:,j]= -X[i,:] + P[j] * X[i,:]
      else:
        dW_one_sample[:,j] = P[j] * X[i,:]
    dW = dW + dW_one_sample

  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  #pass
  num_classes = W.shape[1]
  num_train = X.shape[0]

  f = X.dot(W)                                   # A numpy array of shape (N, C)
  true_distribution = np.zeros_like(f)           # A numpy array of shape (N, C)
  true_distribution[np.arange(num_train), y] = 1 # A numpy array of shape (N, C)
  logC = - np.max(f, axis = 1)                   # A numpy array of shape (1, N)
  logC = np.reshape(logC, (num_train, 1))        # A numpy array of shape (N, 1)
  P = np.exp(f + np.tile(logC, (1, num_classes))) / np.reshape(np.sum(np.exp(f + np.tile(logC, (1, num_classes))), axis = 1), (num_train, 1))  # broadcast, (N, C) / (N, 1), get (N, C) 
  loss = -1 * np.sum(true_distribution * np.log(P))
  loss = loss / num_train + reg * np.sum(W * W)


  dW = dW + (-1 * X.T.dot(true_distribution - P))
  dW = dW / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

