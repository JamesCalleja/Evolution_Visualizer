"""
This module contains neural network-related functions,
primarily activation functions.
"""

import math

def tanh(x):
    return math.tanh(x)

def sigmoid(x): # sigmoid is useful for probability-like outputs (0 to 1)
    return 1 / (1 + math.exp(-1 * x))

def relu(x):
    return max(0, x)

# Note: If you were to create a full NeuralNetwork class, it would reside here.
# For now, creature's brain logic is embedded within the Creature class,
# using these activation functions.