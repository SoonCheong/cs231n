import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ConvNet(object):
  """
  A  convolutional network with the following architecture:
  
  (conv - batchnorm - relu - conv - batchnorm 2x2 max pool)xM - (affine - relu)xN- affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32,32,32], filter_size=[3,3,3],
               hidden_dim=[100], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layers
    - filter_size: Size of filters to use in the convolutional layers
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
    self.num_convs=len(num_filters)
    self.num_affines=len(hidden_dim)
    self.num_layers=self.num_convs*2+self.num_affines+1
    self.filter_size=filter_size
    w_dims=[C]+num_filters
    for i in range(self.num_convs): 
        # conv-batchnorm-relu 1
        j=2*i
        idx_str=str(j+1)
        w_shape=(w_dims[i+1],w_dims[i],filter_size[i],filter_size[i])
        
        self.params['W'+idx_str]=np.random.normal(0,weight_scale,w_shape)
        self.params['b'+idx_str]=np.zeros(num_filters[i])
        self.params['gamma'+idx_str]=np.ones(num_filters[i])
        self.params['beta'+idx_str]=np.zeros(num_filters[i])    
        idx_str=str(j+2)
        # conv-batachnorm-relu-pool 2
        w_shape=(w_dims[i+1],w_dims[i+1],filter_size[i],filter_size[i])
        self.params['W'+idx_str]=np.random.normal(0,weight_scale,w_shape)
        self.params['b'+idx_str]=np.zeros(num_filters[i])
        self.params['gamma'+idx_str]=np.ones(num_filters[i])
        self.params['beta'+idx_str]=np.zeros(num_filters[i])    

    w_dims=[num_filters[self.num_convs-1]*(H/(np.power(2,self.num_convs)))**2]+hidden_dim
    #print np.power(2,self.num_convs)    
    for i in range(self.num_affines): 
    # affine 1
        idx_str=str(i+2*self.num_convs+1)
        self.params['W'+idx_str]=np.random.normal(0, weight_scale, (w_dims[i],w_dims[i+1]))  
        self.params['b'+idx_str]=np.zeros(w_dims[i+1])
        self.params['gamma'+idx_str]=np.ones(w_dims[i+1])
        self.params['beta'+idx_str]=np.zeros(w_dims[i+1])              
        
    # Output layer
    idx_str=str(self.num_layers)
    self.params['W'+idx_str]=np.random.normal(0, weight_scale, (w_dims[self.num_affines],num_classes))
    self.params['b'+idx_str]=np.zeros(num_classes)
    
    self.bn_params = []
    num_batchnorm=self.num_layers-1
    self.bn_params = [{'mode': 'train'} for i in xrange(num_batchnorm)]    
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
    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
        bn_param[mode] = mode
    
    # pass conv_param to the forward pass for the convolutional layer

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_param=[{'stride': 1, 'pad': (size - 1) / 2} for size in self.filter_size]
    conv_cache=[]
    pool_cache=[]
    affine_relu_cache=[]
    h=X
    idx=0
    for i in range(self.num_convs): 
        # conv-batchnorm-relu 1
        idx_str=str(idx+1)

        h, cache=conv_relu_batchnorm_forward(
            h, self.params['W'+idx_str], self.params['b'+idx_str], conv_param[i], 
            self.params['gamma'+idx_str], self.params['beta'+idx_str], self.bn_params[idx])
        conv_cache.append(cache)        
        idx+=1
        
        idx_str=str(idx+1)
        h, cache=conv_relu_batchnorm_forward(
            h, self.params['W'+idx_str], self.params['b'+idx_str], conv_param[i], 
            self.params['gamma'+idx_str], self.params['beta'+idx_str], self.bn_params[idx])
        conv_cache.append(cache)
        idx+=1
        
        h, cache = max_pool_forward_fast(h, pool_param)
        pool_cache.append(cache)
    
    for i in range(self.num_affines):         
        idx_str=str(idx+1)
        h,cache= \
        affine_batchnorm_relu_forward(h, self.params['W'+idx_str], self.params['b'+idx_str],
                                      self.params['gamma'+idx_str], self.params['beta'+idx_str], 
                                      self.bn_params[idx])
        affine_relu_cache.append(cache)
        idx+=1

    
    idx_str=str(self.num_layers)
    scores, affine_cache=affine_forward(h, self.params['W'+idx_str], self.params['b'+idx_str])
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
    reg_loss=0
    for i in range(self.num_layers):
        reg_loss+=np.sum(self.params['W'+str(i+1)]**2)
        
    loss+=0.5*self.reg*reg_loss
    
    dx, dw, db=affine_backward(dx,affine_cache)
    layer_idx=self.num_layers
    idx_str=str(layer_idx)
    grads['W'+idx_str]=dw+self.reg*self.params['W'+idx_str]
    grads['b'+idx_str]=db
    
    for i in range(self.num_affines-1,-1,-1):
        layer_idx-=1
        idx_str=str(layer_idx)
        dx, dw, db, dgamma, dbeta=affine_batchnorm_relu_backward(dx, affine_relu_cache.pop())
        grads['W'+idx_str]=dw+self.reg*self.params['W'+idx_str]
        grads['b'+idx_str]=db
        grads['gamma'+idx_str]=dgamma
        grads['beta'+idx_str]=dbeta
    
    for i in range(self.num_convs-1,-1,-1):
        layer_idx-=1
        idx_str=str(layer_idx)
        dx = max_pool_backward_fast(dx, pool_cache.pop())
        dx, dw, db, dgamma, dbeta=conv_relu_batchnorm_backward(dx, conv_cache.pop())
        grads['W'+idx_str]=dw+self.reg*self.params['W'+idx_str]
        grads['b'+idx_str]=db
        grads['gamma'+idx_str]=dgamma
        grads['beta'+idx_str]=dbeta
    
        layer_idx-=1
        idx_str=str(layer_idx)
        dx, dw, db, dgamma, dbeta=conv_relu_batchnorm_backward(dx, conv_cache.pop())
        grads['W'+idx_str]=dw+self.reg*self.params['W'+idx_str]
        grads['b'+idx_str]=db
        grads['gamma'+idx_str]=dgamma
        grads['beta'+idx_str]=dbeta    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads