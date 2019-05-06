import numpy as np

#### Miscellaneous functions
""" Note that when the input z is a vector or Numpy array, Numpy automatically
applies the function sigmoid elementwise, that is, in vectorized form.  """
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# This 'Layered Chain of Responsibilities' design elegantly catches the duality of algorithms -
# 'feedforward' and 'backpropgation'.
# It is original per my limited knowledge. Thank you JYC taught me Object Oriented Design and 
# introduced me "Design Patterns" in 1999!
# This software is free to use for any purpose given:
#   1. you don't restrict this design's further free distribution and use
#   2. keep this notes as is (you're encouraged to advocate 订阅号: 静安笔记; my wechat: debridge)
# As a Young man, I strive for contribution on AI/deep learning education, fundamental research 
# and more...2019/5/1 20:13:38 
#
# TODO: a non-writable feedforward() like M.N. did for evaluate...

class NeuralLayer:
    def __init__(self, dim_inputs, dim_outputs):
        # this very effective 'little' improvement (/np.sqrt(dim_inputs) is also from
        # http://neuralnetworksanddeeplearning.com
        self.weights = np.random.randn(dim_outputs, dim_inputs)/np.sqrt(dim_inputs) 
        self.biases = np.random.randn(dim_outputs, 1)

    def feedforward(self, inputs, prev_layer):
        if prev_layer is None:
            #this is input layer, set the 'activations'(i.e., outputs) directly as inputs
            self.outputs = inputs
        else:
            self.z = np.dot(self.weights, prev_layer.outputs) + self.biases
            self.outputs = sigmoid(self.z)
        
        self.prev_layer = prev_layer
        return self.outputs

    def backprop(self, next_layer):
        #delta is not exactly the gradient, but very near (c.f. NeuralLayer.update())
        self.delta = (np.dot(next_layer.weights.T, next_layer.delta)) * sigmoid_prime(self.z)
        return self.delta

    def update(self, mini_batch_size, learning_rate):
        self.weights = self.weights -(learning_rate/mini_batch_size)*(np.dot(self.delta, self.prev_layer.outputs.T))
        self.biases = self.biases -(learning_rate/mini_batch_size)*self.delta

# This 'Layered Chain of Responsibilities' design elegantly catches the duality of algorithms -
# 'feedforward' and 'backpropgation'.
# It is original per my limited knowledge. Thank you JYC taught me Object Oriented Design and 
# introduced me "Design Patterns" in 1999!
# This software is free to use for any purpose given:
#   1. you don't restrict this design's further free distribution and use
#   2. keep this notes as is (you're encouraged to advocate 订阅号: 静安笔记; my wechat: debridge)
# As a Young man, I strive for contribution on AI/deep learning education, fundamental research 
# and more...2019/5/1 20:13:38 

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def feedforward(self, x):
        outputs = x.T #depends on the inputs, here flip inputs n*1 array to n,1 vector
        prev_layer = None
        for layer in self.layers:
            outputs = layer.feedforward(outputs, prev_layer)
            prev_layer = layer
        return outputs

    def backprop(self, y):
        # last layer "delta", MSE as cost function
        delta_layer_L = (self.layers[-1].outputs - y) * sigmoid_prime(self.layers[-1].z)
        self.layers[-1].delta = delta_layer_L
        
        # Now let's do the 伟大的反向传播 by 订阅号: 静安笔记 (backprop is here! wechat/微信: debridge)
        next_layer = self.layers[-1]
        for layer in reversed(self.layers[1:-1]):
            delta = layer.backprop(next_layer)
            next_layer = layer

    def update_mini_batch(self, mini_batch, learning_rate):
        for x, y in mini_batch:
            self.feedforward(x)
            self.backprop(y)
        # Updating the weights and biases finally! this is called 梯度下降 (Gradient Descend)
        mini_batch_size = len(mini_batch)
        for layer in reversed(self.layers[1:]):
            layer.update(mini_batch_size, learning_rate)
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate):
        """directly simplified from http://neuralnetworksanddeeplearning.com"""
        """Train the neural network using mini-batch stochastic gradient descent.  
        The ``training_data`` is a list of tuples ``(x, y)`` representing the training inputs 
        and the desired  outputs. """
        print('perform training...')
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            #print("Epoch {0} complete".format(j))
        print('...trained.')
        
    def evaluate(self, test_data, trained=True):
        #TODO: a simplified XOR version for output size == 1 (i.e., ouput layer 1 Neuron)
        model_answers = [self.feedforward(x) for x, y in test_data]
        if trained:
            print("let's evaluate with our __trained__ neural network...")
        else:
            print("let's evaluate with our new neural network...")
        for y_hat, t in zip(model_answers, test_data):
            print(t[0], 'XOR:', y_hat, " = ", t[1])
 

if __name__ == '__main__':
    # faked training data (XOR repeated) to 'fit' SGD framework for future extension
    training_data = [
      [ np.array([[ 0, 0 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 1 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 0 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 1 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 0 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 1 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 0 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 1 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 0 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 1 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 0 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 1 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 0 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 1 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 0 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 1 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 0 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 1 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 0 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 1 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 0 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 1 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 0 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 1 ]]), np.array([[ 0 ]]) ],
    ]

    test_data = [
      [ np.array([[ 0, 0 ]]), np.array([[ 0 ]]) ],
      [ np.array([[ 0, 1 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 0 ]]), np.array([[ 1 ]]) ],
      [ np.array([[ 1, 1 ]]), np.array([[ 0 ]]) ],

    ]

    # This 'Layered Chain of Responsibilities' design elegantly catches the duality of algorithms -
    # 'feedforward' and 'backpropgation'.
    # It is original per my limited knowledge. Thank you JYC taught me Object Oriented Design and 
    # introduced me "Design Patterns" in 1999!
    # This software is free to use for any purpose given:
    #   1. you don't restrict this design's further free distribution and use
    #   2. keep this notes as is (you're encouraged to advocate 订阅号: 静安笔记; my wechat: debridge)
    # As a Young man, I strive for contribution on AI/deep learning education, fundamental research 
    # and more...2019/5/1 20:13:38 

    nn = NeuralNetwork([NeuralLayer(2, 2), 
                        NeuralLayer(2, 16), 
                        NeuralLayer(16, 8),
                        NeuralLayer(8, 1),
                       ])

    nn.evaluate(test_data, trained=False)

    # training starts...
    epochs = 1300
    mini_batch_size = 4
    learning_rate = 3.14
    nn.SGD(training_data, epochs, mini_batch_size, learning_rate)

    # evaluating results...
    nn.evaluate(test_data)
