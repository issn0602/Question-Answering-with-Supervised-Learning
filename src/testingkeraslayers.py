from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input
import numpy as np

class Multiply(Layer):

   def __init__(self, output_dim, **kwargs):
       self.output_dim = output_dim
       super(Multiply, self).__init__(**kwargs)

   def build(self, input_shape):
       # Create a trainable weight variable for this layer.
       self.kernel = self.add_weight(shape=(input_shape[1], self.output_dim),
                                     initializer='uniform',
                                     trainable=True)
       super(Multiply, self).build(input_shape)  # Be sure to call this somewhere!

   def call(self, x):
       return K.dot(x, self.kernel)

   def compute_output_shape(self, input_shape):
       return (input_shape[0], self.output_dim)


#a=K.variable(,dtype='int32')
#print(a.get_shape())
inputs = Input(shape=(2,)) #define the input layer of the network
mult2 = Multiply(output_dim=1)(inputs) #define the second layer of the network
mult = Multiply(output_dim=1)(mult2) #additional layers testing
model = Model(inputs=[inputs],outputs=[mult])

model.compile(optimizer='sgd',loss='mse')


input_data = np.transpose(np.array([[1,2,3,4,5,6,7,8,9,10],[1,1,1,1,1,1,1,1,1,1]]))
#input_data = np.concatenate(input_data,np.transpose(np.ones(10),axes=0))

print(input_data)
output_data = 8 * input_data[:,0]*input_data[:,0]+13 # line equation

model.fit([input_data], [output_data], nb_epoch=10000)
#print(model.get_weights())
