#include <iostream>
#include "mat_lib.hpp"
#include "linear_layer.hpp"
#include "neural_network.hpp"

using std::cout;
using std::endl;

// -------------------- DOT TEST --------------------
void test_dot() {
    cout << "\n=== DOT TEST ===\n";

    MathVec<double> a(3), b(3);
    a[0]=1; a[1]=2; a[2]=3;
    b[0]=4; b[1]=5; b[2]=6;

    cout << "dot = " << (a * b) << endl;
}

// -------------------- MATRIX TEST --------------------
void test_matrix() {
    cout << "\n=== MATRIX TEST ===\n";

    MathMat<double> W(4, 2);

    int v = 1;
    for (size_t i = 0; i < W.rows(); ++i)
        for (size_t j = 0; j < W.cols(); ++j)
            W(i,j) = v++;

    W.print();	
    cout << "W shape: " << W.rows() << " x " << W.cols() << endl;

    auto WT = W.trans();

    WT.print();

    cout << "WT shape: " << WT.rows() << " x " << WT.cols() << endl;
}

// -------------------- MATVEC TEST --------------------
void test_matvec() {
    cout << "\n=== MATVEC TEST ===\n";

    MathMat<double> W(3, 2);

    W(0,0)=1; W(0,1)=2;
    W(1,0)=3; W(1,1)=4;
    W(2,0)=5; W(2,1)=6;

    MathVec<double> x(2);
    x[0]=1;
    x[1]=2;

    auto y = W * x;

    cout << "output size = " << y.size() << endl;
}

// -------------------- TRANSPOSE + BACKWARD CRASH TEST --------------------
void test_backward_crash() {
    cout << "\n=== BACKWARD TEST ===\n";

    LinearLayer<double> layer(2, 3);
    layer.random_init();

    MathVec<double> x(2);
    x[0]=1; x[1]=0;

    auto out = layer.linear(x);

    MathVec<double> grad(out.size());
    for (size_t i = 0; i < grad.size(); ++i)
        grad[i] = 1.0;
    grad.print();	
    cout << "grad size = " << grad.size() << endl;

    auto back = layer.backward(grad);

    cout << "back size = " << back.size() << endl;
}

// -------------------- NETWORK TEST --------------------
void test_network() {
    cout << "\n=== NETWORK TEST ===\n";

    NeuralNetwork<double> net(2, 4, 1);
    net.random_init();

    MathVec<double> x(2);
    x[0]=1; x[1]=1;

    auto y = net.forward(x);

    cout << "output size = " << y.size() << endl;
}

// -------------------- MAIN --------------------
int main() {

    test_dot();
    test_matrix();
    test_matvec();
    test_backward_crash();
    test_network();

    cout << "\nALL TESTS DONE\n";

    return 0;
}
