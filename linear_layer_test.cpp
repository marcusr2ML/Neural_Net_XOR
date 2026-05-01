#include <iostream>
#include "mat_lib.hpp"
#include "linear_layer.hpp"
int main() {

    using T = double;

    // -------------------------
    // Create input vector
    // -------------------------
    MathVec<T> x(4);

    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = -1.0;
    x[3] = 0.5;

    // -------------------------
    // Create linear layer
    // -------------------------
    LinearLayer<T> layer(4, 3);

    // initialize weights and bias randomly
    layer.random_init();    

    // -------------------------
    // Lienar pass
    // -------------------------
    MathVec<T> y = layer.linear(x);

    // -------------------------
    // Print output
    // -------------------------
    std::cout << "Output vector:\n";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << y[i] << "\n";
    }

    // -------------------------
    // forward pass
    // -------------------------
    y = layer.ReLU(y);

    // -------------------------
    // Print output
    // -------------------------
    std::cout << "Output vector:\n";
    for (size_t i = 0; i < 3; ++i) {
        std::cout << y[i] << "\n";
    }


    return 0;
}
