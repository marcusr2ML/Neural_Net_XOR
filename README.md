# Neural_Net_XOR

A minimal neural network implementation in C++ built from scratch using a custom linear algebra library. Includes a fully connected feedforward neural network trained on the XOR problem.
**NOTE: this readme and the script tests were AI generated**
---

# 1. Linear Algebra Library (MathVec / MathMat)

This project implements a lightweight linear algebra system with vector and matrix wrappers.

## 1.1 MathVec Operations

### Construction
- MathVec<T>() → empty vector
- MathVec<T>(dim) → zero-initialized vector
- MathVec<T>({a, b, c}) → initializer list

### Element Access
- v[i] → read/write access (bounds checked)

### Vector Operations

- Addition: v1 + v2
- Dot product: v1 * v2
- Outer product: v1 ^ v2
- Elementwise product: v1 % v2
- Scalar multiplication: a * v or v * a
- In-place addition: v1 += v2
- Random init: v.random_init(scale)

---

## 1.2 MathMat Operations

### Construction
- MathMat<T>(rows, cols)

### Element Access
- M(i, j)

### Matrix Operations

- Addition: A + B
- Multiplication: A * B
- Matrix-vector: A * v
- In-place addition: A += B
- Transpose: A.trans()
- Random init: M.random_init(scale)

---

# 2. Linear Layer

Fully connected layer with:
- ReLU / Sigmoid activation
- Backpropagation
- Gradient accumulation

## Parameters
- weights (out × in)
- bias (out)

## Forward Pass
- z = W x + b
- activation(z)

ReLU:
- max(0, x)

Sigmoid:
- 1 / (1 + exp(-x))

---

## Backward Pass

Given grad_out:

- Apply activation derivative
- Accumulate gradients:
  - grad_b += grad
  - grad_W += grad ⊗ input
- Backprop:
  - W^T * grad

---

## Update Step

- W -= lr * grad_W
- b -= lr * grad_b

---

## Zero Grad

Resets all gradients to 0.

---

# 3. Neural Network

Two-layer MLP:

Input → Hidden(ReLU) → Output(Sigmoid)

## Forward
- l1(x)
- l2(l1(x))

## Backward
- l2 backward
- l1 backward

## Update
- step both layers

---

# 4. XOR Training

Dataset:
- (0,0)->0
- (0,1)->1
- (1,0)->1
- (1,1)->0

Loss:
- MSE = (pred - target)^2

Gradient:
- grad = pred - target

Training loop:
- forward all samples
- backward accumulate
- one update per epoch

---

# 5. Build

g++ main.cpp -O2 -o xor_net
./xor_net

---

# 6. Result

Network learns XOR:

0 XOR 0 = 0  
0 XOR 1 = 1  
1 XOR 0 = 1  
1 XOR 1 = 0  

---
