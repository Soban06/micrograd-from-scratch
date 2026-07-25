# Micrograd from Scratch

A lightweight, scalar-valued automatic differentiation (autograd) engine and dynamic computational graph implementation built from pure Python, inspired by Andrej Karpathy's *Neural Networks: Zero to Hero* series.

## Features
- **Autograd Engine:** Implements reverse-mode automatic differentiation over a Directed Acyclic Graph (DAG).
- **Core Operations:** Supports addition, multiplication, exponentiation, negation, subtraction, division, and `tanh`/`exp` activations.
- **Neural Network Primitives:** Custom `Neuron`, `Layer`, and Multi-Layer Perceptron (`MLP`) modules built on top of `Value` scalar nodes.
- **Graph Visualization:** Built-in integration with Graphviz to render computational nodes, values, operations, and backpropagated gradients in real time.

## Quickstart

```python
from engine import Value

# Define scalar operations
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = a * b; c.label = 'c'

# Backpropagate
c.backward()

print(f"dc/da: {a.grad}") # -3.0
print(f"dc/db: {b.grad}") #  2.0
