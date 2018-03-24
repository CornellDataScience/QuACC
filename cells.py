"""
Custom RNNCell Class
"""

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import math_ops


class GatedAttentionCell(RNNCell):
    """Gated Attention cell

    Attributes:
    num_units (int):                  the number of units in the cell.
    activation (func):                activation function, use tanh by default
    reuse (bool):                     reuse variables or not
    kernel_initializer (initializer): the initializer for the weight matrices
    bias_initializer (initializer):   the initializer for the bias.
    name (str):                       the name of the layer
    """
    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None,
                 name=None):
        super(GatedAttentionCell, self).__init__()
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    # giving property to expose hidden attributes
    # don't change here
    @property
    def state_size(self):
        return self._num_units

    # don't change here
    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope = None):
        """ Run this Attention cell on inputs from the given state.
        Args:
            inputs (tf.Tensor):          must have shape [batch_size, input_size]
            state (tf.Tensor or tuple):  if self.state_size is an integer, this should be a 2-D Tensor with shape
                                         [batch_size, self.state_size].
                                         Otherwise, if self.state_size is a tuple of integers, this should be a tuple
                                         with shapes [batch_size, s] for s in self.state_size
            scope (str):                 scope of variables

        Returns:
            output (tf.Tensor):          a tensor with shape [batch_size, self.output_size]
            new_state (tf.Tensor):       should have the same shape with input state
        """
        output = None
        new_state = None
        return output, new_state
