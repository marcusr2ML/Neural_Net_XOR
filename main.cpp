#include <iostream>
#include <vector>
#include "neural_network.hpp"

// XOR dataset
struct Sample {
    MathVec<double> x;
    MathVec<double> y;
};

int main() {

    // -------------------------
    // 1. Build XOR dataset
    // -------------------------
    std::vector<Sample> data;

    {
        MathVec<double> x(2), y(1);

        x[0]=0; x[1]=0; y[0]=0;
        data.push_back({x, y});

        x[0]=0; x[1]=1; y[0]=1;
        data.push_back({x, y});

        x[0]=1; x[1]=0; y[0]=1;
        data.push_back({x, y});

        x[0]=1; x[1]=1; y[0]=0;
        data.push_back({x, y});
    }

    // -------------------------
    // 2. Create network
    // -------------------------
    NeuralNetwork<double> net(2, 8, 1);
    net.random_init(.01);

    double lr = 0.1;

    // -------------------------
    // 3. Training loop (FIXED)
    // -------------------------
    for (int epoch = 0; epoch < 10000; ++epoch) {

        double total_loss = 0.0;
	net.zero_grad();

        for (auto &s : data) {

            // forward
            MathVec<double> pred = net.forward(s.x);

            // loss derivative (MSE)
            MathVec<double> grad(1);
            grad[0] = pred[0] - s.y[0];

            total_loss += grad[0] * grad[0]; 

            // backward (accumulates gradients internally)
            net.backward(grad); //I absorbed the additional factor of 2 into the learning rate
        }

        //ONE update per epoch
        net.step(lr);

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch
                      << " Loss: " << total_loss << "\n";
        }
    }

    // -------------------------
    // 4. Test results
    // -------------------------
    std::cout << "\nFinal predictions:\n";




    for (auto &s : data) {
        MathVec<double> pred = net.forward(s.x);

        std::cout << s.x[0] << " XOR " << s.x[1]
                  << " = " << std::round(pred[0])
                  << " (target " << s.y[0] << ")\n";
    }

    return 0;
}
