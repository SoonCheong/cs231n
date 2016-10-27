import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32,16], filter_size=[7,5],
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W=input_dim
    # conv-relu-pool 1
    self.params['W1']=np.random.normal(0,weight_scale,(num_filters[0],C,filter_size[0],filter_size[0]))
    self.params['b1']=np.zeros(num_filters[0])

    # conv-relu-pool 2
    w2_dim0=num_filters[0]*(H/2)**2
    self.params['W2']=np.random.normal(0,weight_scale,(num_filters[1],num_filters[0],filter_size[1],filter_size[1]))
    self.params['b2']=np.zeros(num_filters[1])
    
    # affine 1
    w3_dim0=num_filters[1]*(H/4)**2
    self.params['W3']=np.random.normal(0, weight_scale, (w3_dim0,hidden_dim))            
    self.params['b3']=np.zeros(hidden_dim)
              
    # affine 2
    self.params['W4']=np.random.normal(0, weight_scale, (hidden_dim,num_classes))
    self.params['b4']=np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = [W1.shape[2], W2.shape[2]]
    conv_param1 = {'stride': 1, 'pad': (filter_size[0] - 1) / 2}
    conv_param2 = {'stride': 1, 'pad': (filter_size[1] - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    h1,conv_cache1=conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)
    h2,conv_cache2=conv_relu_pool_forward(h1, W2, b2, conv_param2, pool_param)
    h3,affine_relu_cache=affine_relu_forward(h2, W3, b3)
    scores, affine_cache=affine_forward(h3, W4, b4)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################

    loss, dx=softmax_loss(scores, y)
    loss+=0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2)+np.sum(W4**2))
    
    dx, dw, db=affine_backward(dx,affine_cache)
    grads['W4']=dw+self.reg*W4
    grads['b4']=db
    
    dx, dw, db=affine_relu_backward(dx, affine_relu_cache)
    grads['W3']=dw+self.reg*W3
    grads['b3']=db
    
    dx, dw, db=conv_relu_pool_backward(dx, conv_cache2)
    grads['W2']=dw+self.reg*W2
    grads['b2']=db    
    
    dx, dw, db=conv_relu_pool_backward(dx, conv_cache1)
    grads['W1']=dw+self.reg*W1
    grads['b1']=db        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
