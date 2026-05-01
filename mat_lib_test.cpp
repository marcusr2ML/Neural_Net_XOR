#include <iostream>
#include "mat_lib.hpp"

int main() {

    //================ VECTOR TEST =================//
    MathVec<double> a(3);
    MathVec<double> b(3);

    a[0] = 1.0; a[1] = 2.0; a[2] = 3.0;
    b[0] = 4.0; b[1] = 5.0; b[2] = 6.0;

    for (int i = 0; i < 3; ++i)
        std::cout << a[i] << " ";
    for (int i = 0; i < 3; ++i)
        std::cout << b[i] << " ";


    auto c = a + b;
    auto dot = a * b;

    std::cout << "Vector result: ";
    for (int i = 0; i < 3; ++i)
        std::cout << c[i] << " ";

    std::cout << "\nDot product: " << dot << "\n";


    //================ MATRIX TEST =================//
    MathMat<double> A(2, 2);
    MathMat<double> B(2, 2);

    A(0,0) = 1; A(0,1) = 2;
    A(1,0) = 3; A(1,1) = 4;

    B(0,0) = 5; B(0,1) = 6;
    B(1,0) = 7; B(1,1) = 8;

    auto C = A + B;
    auto D = A * B;

    std::cout << "\nMatrix addition:\n";
    std::cout << C(0,0) << " " << C(0,1) << "\n";
    std::cout << C(1,0) << " " << C(1,1) << "\n";

    std::cout << "\nMatrix multiplication:\n";
    std::cout << D(0,0) << " " << D(0,1) << "\n";
    std::cout << D(1,0) << " " << D(1,1) << "\n";

    MathMat<double> outer = a ^ b;

    std::cout << "Outer product matrix:\n";

    for (size_t i = 0; i < outer.rows(); ++i) {
        for (size_t j = 0; j < outer.cols(); ++j) {
            std::cout << outer(i, j) << " ";
        }
        std::cout << "\n";
    }


    return 0;
}
