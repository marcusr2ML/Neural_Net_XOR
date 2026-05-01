#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>

using std::vector;
using std::cout;
using std::endl;

//--FORWARD DECLARATIONS

template<typename T> class MathVec;
template<typename T> class MathMat;

template<typename T> MathVec<T> operator*(T scalar, const MathVec<T>& v);
template<typename T> MathVec<T> operator*(const MathVec<T>& v, T scalar);

template<typename T> MathVec<T> vec_add(const MathVec<T>& a, const MathVec<T>& b);
template<typename T> T vec_dot(const MathVec<T>& a, const MathVec<T>& b);
template<typename T> MathMat<T> vec_outer(const MathVec<T>& a, const MathVec<T>& x);
template<typename T> MathVec<T> vec_ew_prod(const MathVec<T>& a, const MathVec<T>& x);
template<typename T> MathVec<T>& vec_iadd(MathVec<T>& a, const MathVec<T>& b);
template<typename T> void vec_random_init(MathVec<T>& v, T scale);

template<typename T> MathMat<T> mat_add(const MathMat<T>& a, const MathMat<T>& b);
template<typename T> MathMat<T> mat_mul(const MathMat<T>& a, const MathMat<T>& b);
template<typename T> MathVec<T> mat_vec(const MathMat<T>& a, const MathVec<T>& x);
template<typename T> MathMat<T>& mat_iadd(MathMat<T>& a, const MathMat<T>& b);
template<typename T> void mat_random_init(MathMat<T>& m, T scale);

template <typename T>
T random_value();


//VECTOR WRAPPER-//
template <typename T>
class MathVec{

	private:
		vector<T> vec;		
	public:
		MathVec(){}
		MathVec(int dim):vec(dim){}
		MathVec(std::initializer_list<T> init):vec(init){}

	size_t size() const { return vec.size(); }

	bool empty() const { return vec.empty(); }

    //--DEFINE OVERLEADED OPS
	//indexing
   	 T& operator[](size_t i) {
        	return vec[i];
    	}

    	const T& operator[](size_t i) const {
        	return vec[i];	
	}

	//scalar-vec mult
	friend MathVec<T> operator*<>(T scalar, const MathVec<T>& v);
	friend MathVec<T> operator*<>(const MathVec<T>& v, T scalar);

	//vec-vec mult
	MathVec<T> operator+(const MathVec<T>& vec2) const{
		return vec_add(*this, vec2);
	}		

	T operator*(const MathVec<T>& vec2) const {
    		return vec_dot(*this, vec2);
	}	

	MathMat<T> operator^(const MathVec<T>& vec2) const{
		return vec_outer(*this, vec2);
	}

        MathVec<T> operator%(const MathVec<T>& vec2) const{
                return vec_ew_prod(*this, vec2);
        }

	MathVec<T>& operator+=(const MathVec<T>& vec2){
		return vec_iadd(*this, vec2);
	}

   //--INIT RANDOM VALUES	
        void random_init(T scale = T(1)) {
                vec_random_init(*this, scale);
        }

   //--PRINT FUNCTIONS
	void print() const{
		for(size_t i = 0; i<size();i++){
			cout<<","<< vec[i];
		}
		cout<<endl;
	}
};


//--MATRIX WRAPPER--//
template <typename T>
class MathMat{

        private:
                vector<vector<T>> mat;

        public:
                MathMat() {}

                MathMat(size_t rows, size_t cols)
                        : mat(rows, vector<T>(cols)) {}

	size_t rows() const {
    		return mat.size();
	}

	size_t cols() const {
		if (mat.empty()) return 0;
    		size_t c = mat[0].size();

    		for (const auto& row : mat) {
        		if (row.size() != c)
            			throw std::runtime_error("MathMat: inconsistent row sizes");
    		}

    		return c;
	}

	bool empty() const {
		return mat.empty();
	}

        T& operator()(size_t i, size_t j) {
                return mat[i][j];
        }

        const T& operator()(size_t i, size_t j) const {
                return mat[i][j];
        }

        MathMat<T> operator+(const MathMat<T>& mat2) const {
                return mat_add(*this, mat2);
        }

	MathMat<T> operator*(const MathMat<T>& mat2) const {
        	return mat_mul(*this, mat2);
	}	

	MathVec<T> operator*(const MathVec<T>& vec) const {
    		return mat_vec(*this, vec);
	}

	MathMat<T>& operator+=(const MathMat<T>& mat2){
		return mat_iadd(*this, mat2);
	}

	//--TRANSPOSE MATRIX
	MathMat<T> trans() const {
		MathMat<T> result(cols(), rows());

    		for (size_t i = 0; i < rows(); ++i)
        		for (size_t j = 0; j < cols(); ++j)
            			result(j, i) = mat[i][j];

    		return result;
	}

   //--INIT RANDOM VALUES	
        void random_init(T scale = T(1)) {
                mat_random_init(*this, scale);
        }

   //--PRINT FUNCTIONS
        void print() const{
                if(rows() == 0 || cols() == 0){
                        cout<<"[ERROR] empty matrix"<<endl;
                        return;
                }

                for(size_t i = 0; i<rows(); i++){
                        for(size_t j = 0; j<cols(); j++){
                                cout<<","<< (*this)(i,j);
                        }
                        cout<<endl;
                }
        }
};

//====================================
//FUNCTIONS
//====================================

//MATVEC FUNCTIONS
template<typename T>
MathVec<T> vec_add(const MathVec<T>& a, const MathVec<T>& b) {
    if (a.empty() || b.empty())
        throw std::runtime_error("MathVec: empty vector in +");
    if (a.size() != b.size())
        throw std::runtime_error("MathVec: size mismatch in +");

    MathVec<T> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}

template<typename T>
T vec_dot(const MathVec<T>& a, const MathVec<T>& b) {
    if (a.empty() || b.empty())
        throw std::runtime_error("MathVec: empty vector in dot");
    if (a.size() != b.size())
        throw std::runtime_error("MathVec: size mismatch in dot");

    T result{};
    for (size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];
    return result;
}

template<typename T>
MathVec<T> operator*(T scalar, const MathVec<T>& v){

	if (v.empty())
		throw std::runtime_error("MathVec: empty vector in scalar mult");

	MathVec<T> result(v.size());

	for (size_t i = 0; i < v.size(); ++i)
		result[i] = scalar * v[i];

	return result;
}

template<typename T>
MathVec<T> operator*(const MathVec<T>& v, T scalar){

	if (v.empty())
		throw std::runtime_error("MathVec: empty vector in scalar mult");

	MathVec<T> result(v.size());

	for (size_t i = 0; i < v.size(); ++i)
		result[i] = v[i] * scalar;

	return result;
}


template<typename T>
void vec_random_init(MathVec<T>& v, T scale) {
    if (v.empty())
        throw std::runtime_error("MathVec: cannot init empty vector");

    for (size_t i = 0; i < v.size(); ++i)
        v[i] = random_value<T>()*scale;
}

template<typename T>
MathMat<T> vec_outer(const MathVec<T>& a, const MathVec<T>& b) {
    if (a.empty() || b.empty())
        throw std::runtime_error("MathVec: empty vector in outer");

    MathMat<T> result(a.size(), b.size());

    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            result(i,j) = a[i]*b[j];

    return result;
}

template<typename T>
MathVec<T> vec_ew_prod(const MathVec<T>& a, const MathVec<T>& b) {
    if (a.empty() || b.empty())
        throw std::runtime_error("MathVec: empty vector in elementwise prod");
    if (a.size() != b.size())
        throw std::runtime_error("MathVec: size mismatch in elementwise prod");

    MathVec<T> result(a.size());

    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i]*b[i];

    return result;
}

template<typename T>
MathVec<T>& vec_iadd(MathVec<T>& a, const MathVec<T>& b) {
    if (a.empty() || b.empty())
        throw std::runtime_error("MathVec: empty vector in +=");
    if (a.size() != b.size())
        throw std::runtime_error("MathVec: size mismatch in +=");

    for (size_t i = 0; i < a.size(); ++i)
        a[i] += b[i];

    return a;
}



//-MATHMAT FUNCTIONS
template<typename T>
MathMat<T> mat_add(const MathMat<T>& a, const MathMat<T>& b) {
    if (a.empty() || b.empty())
        throw std::runtime_error("MathMat: empty matrix in +");
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::runtime_error("MathMat: size mismatch in +");

    MathMat<T> result(a.rows(), a.cols());

    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            result(i,j) = a(i,j) + b(i,j);

    return result;
}

template<typename T>
MathMat<T> mat_mul(const MathMat<T>& a, const MathMat<T>& b) {
    if (a.empty() || b.empty())
        throw std::runtime_error("MathMat: empty matrix in *");
    if (a.cols() != b.rows())
        throw std::runtime_error("MathMat: size mismatch in *");

    MathMat<T> result(a.rows(), b.cols());

    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < b.cols(); ++j) {
            T sum{};
            for (size_t k = 0; k < a.cols(); ++k)
                sum += a(i,k) * b(k,j);
            result(i,j) = sum;
        }
    }

    return result;
}

template<typename T>
MathVec<T> mat_vec(const MathMat<T>& a, const MathVec<T>& x) {
    if (a.empty())
        throw std::runtime_error("MathMat: empty matrix in mat_vec");
    if (x.empty())
        throw std::runtime_error("MathVec: empty vector in mat_vec");
    if (a.cols() != x.size())
        throw std::runtime_error("Dimension mismatch");

    MathVec<T> result(a.rows());

    for (size_t i = 0; i < a.rows(); ++i) {
        T sum{};
        for (size_t j = 0; j < a.cols(); ++j)
            sum += a(i,j) * x[j];
        result[i] = sum;
    }

    return result;
}

template<typename T>
MathMat<T>& mat_iadd(MathMat<T>& a, const MathMat<T>& b) {
    if (a.empty() || b.empty())
        throw std::runtime_error("MathMat: empty matrix in +=");
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::runtime_error("MathMat: size mismatch in +=");

    for (size_t i = 0; i < a.rows(); ++i)
        for (size_t j = 0; j < a.cols(); ++j)
            a(i,j) += b(i,j);

    return a;
}

template<typename T>
void mat_random_init(MathMat<T>& m, T scale) {
    if (m.empty())
        throw std::runtime_error("MathMat: cannot init empty matrix");

    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j)
            m(i,j) = random_value<T>() * scale;
}


//-RNG
template <typename T>
T random_value() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<T> dist(0.0, 1.0);
    return dist(gen);
}
