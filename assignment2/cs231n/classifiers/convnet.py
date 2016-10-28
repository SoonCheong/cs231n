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
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[64,32,16,4], filter_size=[7,5,3,3],
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
    # conv-batchnorm-relu 1
    self.params['W1']=np.random.normal(0,weight_scale,(num_filters[0],C,filter_size[0],filter_size[0]))
    self.params['b1']=np.zeros(num_filters[0])
    self.params['gamma1']=np.ones(num_filters[0])
    self.params['beta1']=np.zeros(num_filters[0])    

    # conv-batachnorm-relu-pool 2
    #w2_dim0=num_filters[0]*(H/2)**2
    self.params['W2']=np.random.normal(0,weight_scale,(num_filters[1],num_filters[0],filter_size[1],filter_size[1]))
    self.params['b2']=np.zeros(num_filters[1])
    self.params['gamma2']=np.ones(num_filters[1])
    self.params['beta2']=np.zeros(num_filters[1])    
    
    self.params['W3']=np.random.normal(0,weight_scale,(num_filters[2],num_filters[1],filter_size[2],filter_size[2]))
    self.params['b3']=np.zeros(num_filters[2])
    self.params['gamma3']=np.ones(num_filters[2])
    self.params['beta3']=np.zeros(num_filters[2])    

    # conv-batachnorm-relu-pool 2
    #w2_dim0=num_filters[0]*(H/2)**2
    self.params['W4']=np.random.normal(0,weight_scale,(num_filters[3],num_filters[2],filter_size[3],filter_size[3]))
    self.params['b4']=np.zeros(num_filters[3])
    self.params['gamma4']=np.ones(num_filters[3])
    self.params['beta4']=np.zeros(num_filters[3])    
    
    # affine 1
    w3_dim0=num_filters[3]*(H/4)**2
    print w3_dim0
    self.params['W5']=np.random.normal(0, weight_scale, (w3_dim0,hidden_dim))            
    self.params['b5']=np.zeros(hidden_dim)
              
    # affine 2
    self.params['W6']=np.random.normal(0, weight_scale, (hidden_dim,num_classes))
    self.params['b6']=np.zeros(num_classes)
    
    self.bn_params = []
    num_batchnorm=4
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
    
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    W6, b6 = self.params['W6'], self.params['b6']    
    gamma1, beta1=self.params['gamma1'], self.params['beta1']
    gamma2, beta2=self.params['gamma2'], self.params['beta2']
    gamma3, beta3=self.params['gamma3'], self.params['beta3']
    gamma4, beta4=self.params['gamma4'], self.params['beta4']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = [W1.shape[2], W2.shape[2], W3.shape[2], W4.shape[2]]
    conv_param1 = {'stride': 1, 'pad': (filter_size[0] - 1) / 2}
    conv_param2 = {'stride': 1, 'pad': (filter_size[1] - 1) / 2}
    conv_param3 = {'stride': 1, 'pad': (filter_size[2] - 1) / 2}
    conv_param4 = {'stride': 1, 'pad': (filter_size[3] - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    h1, conv_cache1=conv_relu_batchnorm_forward(X, W1, b1, conv_param1, gamma1, beta1, self.bn_params[0])
    h2, conv_cache2=conv_relu_batchnorm_forward(h1, W2, b2, conv_param2, gamma2, beta2, self.bn_params[1])    
    h3, pool_cache1 = max_pool_forward_fast(h2, pool_param)
    
    h4, conv_cache3=conv_relu_batchnorm_forward(h3, W3, b3, conv_param3, gamma3, beta3, self.bn_params[2])
    h5, conv_cache4=conv_relu_batchnorm_forward(h4, W4, b4, conv_param4, gamma4, beta4, self.bn_params[3])    
    h6, pool_cache2 = max_pool_forward_fast(h5, pool_param)
    
    h7,affine_relu_cache=affine_relu_forward(h6, W5, b5)
    scores, affine_cache=affine_forward(h7, W6, b6)
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
    loss+=0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2)+
                        np.sum(W3**2)+np.sum(W4**2)+
                        np.sum(W5**2)+np.sum(W6**2))
    
    dx, dw, db=affine_backward(dx,affine_cache)
    grads['W6']=dw+self.reg*W6
    grads['b6']=db
    
    dx, dw, db=affine_relu_backward(dx, affine_relu_cache)
    grads['W5']=dw+self.reg*W5
    grads['b5']=db

    dx = max_pool_backward_fast(dx, pool_cache2)
    
    dx, dw, db, dgamma, dbeta=conv_relu_batchnorm_backward(dx, conv_cache4)
    grads['W4']=dw+self.reg*W4
    grads['b4']=db    
    grads['gamma4']=dgamma
    grads['beta4']=dbeta
    
    dx, dw, db, dgamma, dbeta=conv_relu_batchnorm_backward(dx, conv_cache3)
    grads['W3']=dw+self.reg*W3
    grads['b3']=db      
    grads['gamma3']=dgamma
    grads['beta3']=dbeta      
    
    dx = max_pool_backward_fast(dx, pool_cache1)
    
    dx, dw, db, dgamma2, dbeta2=conv_relu_batchnorm_backward(dx, conv_cache2)
    grads['W2']=dw+self.reg*W2
    grads['b2']=db    
    grads['gamma2']=dgamma2
    grads['beta2']=dbeta2
    
    dx, dw, db, dgamma1, dbeta1=conv_relu_batchnorm_backward(dx, conv_cache1)
    grads['W1']=dw+self.reg*W1
    grads['b1']=db      
    grads['gamma1']=dgamma1
    grads['beta1']=dbeta1    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
