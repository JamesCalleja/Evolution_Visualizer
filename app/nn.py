"""
This module contains neural network-related functions,
primarily activation functions.
"""

import math

def tanh(value):
    """Hyperbolic tangent activation function."""
    return math.tanh(value)

def sigmoid(value):
    """Sigmoid activation function (useful for probability-like outputs)."""
    return 1 / (1 + math.exp(-1 * value))

def relu(value):
    """ReLU (Rectified Linear Unit) activation function."""
    return max(0, value)

# Note: If you were to create a full NeuralNetwork class, it would reside here.
# For now, creature's brain logic is embedded within the Creature class,
# using these activation functions.