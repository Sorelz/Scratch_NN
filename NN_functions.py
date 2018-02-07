import numpy as np

#initializing parameters
def initialize_parameters_deep(layers,seed):
    '''
    layers: list containing the number of neurons by layer (including input) 
    '''
    np.random.seed(seed)
    params = {}  
    print("{} Layers parameters init".format(len(layers)))

    for l in range(1, len(layers)):
        #small values initialization for faster updating of weights
        params["W" + str(l)] = np.random.randn(layers[l],layers[l-1])*np.sqrt(2/layers[l-1]) #He initialization
        params["b" + str(l)] = np.zeros((layers[l],1))
        #control the dimensions is working
        assert(params["W" + str(l)].shape == (layers[l], layers[l-1]))
        assert(params['b' + str(l)].shape == (layers[l],1))

    return params  

#some activations functions
def sigmoid(z):
    '''
    element-wise sigmoid 
    '''
    sig = 1/(1+np.exp(-z))
    return sig
def relu(z):
    '''
    element-wise relu
    '''
    rel=np.maximum(0,z)
    return rel
#their derivatives
def relu_deriv(z):
    x=np.copy(z)
    x[x<=0] = 0
    x[x>0] = 1
    return x
def sigmoid_deriv(z):
    x=sigmoid(z)*(1-sigmoid(z))
    return x

#Forward step
def forward(A, W, b, activation="relu",keep_prob=1,init="no"):
    '''
    Implement one step of forward
    cache: contains A,W,b,Z; stored for computing the backward pass efficiently
    '''
    D = np.random.rand(A.shape[0],A.shape[1])
    if(init=="1"):
        1
    else:         
        A = A*(D < keep_prob)      
        A = np.divide(A,keep_prob) 
    
    Z = np.dot(W,A)+b
    if activation == "sigmoid":
        A_new = sigmoid(Z)
    elif activation == "relu":
        A_new = relu(Z)

    cache = ((A,W,b),Z,D)
    return A_new, cache

def prop_forward(X, params, activation="relu",keep_prob=1):
    '''
    Implement forward propagation for the same activation for all layers, and finalizing with sigmoid
    '''
    caches = []
    A = X
    L = len(params)//2 #contains Wi,bi for each layer i so divide by 2.
    
    for l in range(1, L):
        A_old = A
        A, cache = forward(A_old,params["W"+str(l)],params["b"+str(l)],activation,keep_prob,str(l))
        caches.append(cache)

    A_pred, cache = forward(A,params["W"+str(L)],params["b"+str(L)],"sigmoid",keep_prob)
    caches.append(cache)
    return A_pred, caches

#Cost step
def compute_cost(A_pred,y,params,lambd=0):
    '''
    cross-entropy cost with regularization possible
    '''
    n_W=len(params)//2
    weights_decay=0
    regu_L2=0
    m=y.shape[1]
    if lambd!=0:
        for i in range(n_W):
            weights_decay+=np.sum(np.square(params["W"+str(i+1)]))
        regu_L2=(lambd/(2*m))*weights_decay
    cost = -(1/m)*(np.dot(np.log(A_pred),y.T)+np.dot(np.log(1-A_pred),(1-y).T))+regu_L2
    cost = np.squeeze(cost)   
    return cost

#Backward Step
def relu_dZ(dA,z_cache):
    dZ=dA*relu_deriv(z_cache)
    return dZ
def sigmoid_dZ(dA,z_cache):
    dZ=dA*sigmoid_deriv(z_cache)
    return dZ

def sub_back(dZ, cache,d_cache,lambd=0,keep_prob=1):
    """
    Compute DWl,dbl,DA_old based on dZl
    """
    A_old, W, b = cache
    m = A_old.shape[0]

    dW = (1/m)*np.dot(dZ,A_old.T)+(lambd/m)*W
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_old = (np.dot(W.T,dZ)*d_cache)/keep_prob
    
    return dA_old, dW, db

def backward(dA, cache, activation,lambd=0,keep_prob=1):
    """
    compute one-step of backward prop
    """
    awb_cache, z_cache,d_cache = cache
    
    if activation == "relu":
        dZ = relu_dZ(dA,z_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_dZ(dA,z_cache)
        
    dA_prev, dW, db = sub_back(dZ,awb_cache,d_cache,lambd,keep_prob)
    
    return dA_prev, dW, db

def prop_backward(AL, Y, caches,activation,lambd=0, keep_prob=1):
    
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))    
    current_cache = caches[L-1]
    grads["dA"+str(L)], grads["dW"+str(L)], grads["db"+str(L)] = backward(dAL,current_cache,"sigmoid",lambd,keep_prob)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward(grads["dA"+str(l+2)],current_cache,activation,lambd,keep_prob)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

#updating parameters via gradient descent
def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 

    for l in range(1,L+1):
        parameters["W"+str(l)]=parameters["W"+str(l)]-learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)]=parameters["b"+str(l)]-learning_rate*grads["db"+str(l)]
        
    return parameters

