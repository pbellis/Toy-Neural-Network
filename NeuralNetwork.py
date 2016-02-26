import numpy as np

class NeuralNetwork(object):
    
    def __init__(self, layer_dims, epsilon, activation_function, l2_lambda = 0):
        self.layer_dims = layer_dims
        self.epsilon = epsilon
        self.activation_function = activation_function
        
        self.layers = len(self.layer_dims)
        
        self.input_layer= 0
        self.output_layer = self.layers - 1
        
        self.W = [np.random.randn(self.layer_dims[j], self.layer_dims[j+1]) / np.sqrt(self.layer_dims[j]) for j in xrange(self.output_layer)]
        self.b = [np.zeros((1, layer_dim)) for layer_dim in self.layer_dims[1:]]
        
        self.l2_lambda = l2_lambda
        
    def foward_propagate(self, X):
        activations = [None for layer_dim in self.layer_dims]
        activations[self.input_layer] = X
        
        for l in xrange(self.input_layer+1, self.output_layer):
            zl = activations[l-1].dot(self.W[l-1]) + self.b[l-1]
            activations[l] = self.activation_function(zl)
            
        zk = activations[self.output_layer - 1].dot(self.W[self.output_layer - 1]) + self.b[self.output_layer - 1]
        exp_zk = np.exp(zk)
        activations[self.output_layer] = exp_zk / np.sum(exp_zk, axis=1, keepdims=True)
        
        return activations
        
    def back_propagate(self, X, y, activations):                
        num_samples = len(X)

        beta  = [None for i in xrange(self.layers)]

        dW = [None for j in xrange(self.output_layer)]
        db = [None for i in xrange(self.output_layer)]

        beta[self.output_layer] = activations[self.output_layer]
        beta[self.output_layer][xrange(num_samples), y] -= 1        
                        
        for k in xrange(self.output_layer, self.input_layer, -1):
            beta[k-1] = (beta[k].dot(self.W[k-1].T) * self.activation_function(activations[k-1], True))
            
            l2 = self.l2_lambda * np.sqrt( np.sum( np.square(self.W[k-1] ) ) ) / self.W[k-1].size
            
            dW[k-1] = self.epsilon * (activations[k-1].T).dot(beta[k]) + l2 * self.W[k-1]
            db[k-1] = self.epsilon * np.sum(beta[k], axis=0, keepdims=True)
            
        return dW, db
        
    def predict(self, X):
        return np.argmax(self.foward_propagate(X)[self.output_layer], axis=1)
        
    def fit(self,X,y,num_epochs):        
        for epoch in xrange(num_epochs):
            activations = self.foward_propagate(X)
            dW, db = self.back_propagate(X, y, activations)
            
            for l in xrange(self.output_layer):
                self.W[l] -= dW[l]
                self.b[l] -= db[l]
                
    def compute_cost(self, X, y):
        num_samples = len(X)

        softmax_scores = self.foward_propagate(X)[self.output_layer]
        cross_ent_err = -np.log(softmax_scores[range(num_samples), y])
        data_loss = np.sum(cross_ent_err)
        return 1./num_samples * data_loss
        
