
#include <cstdio>
#include <vector>

using std::vector;

//VECTOR WRAPPER-//
template <typename T>
class MathVec{

	private:
		vector<T> vec;		
	public:
		MathVec(){}
		MathVec(int dim):vec(dim){}


    //--DEFINE OVERLEADED OPS
    	// vector-index: can modify
   	 T& operator[](size_t i) {
        	return vec[i];
    	}

    	// vector-index: read-only
    	const T& operator[](size_t i) const {
        	return vec[i];	
	}

	// vector-addition
	MathVec<T> operator+(const MathVec<T>& vec2) const{
		MathVec<T> result(vec.size());		
		for (size_t i=0; i<vec.size(); ++i){
			result[i] = vec[i]+vec2[i];
		}
		return result;
	}	

        // vector-mult: dot-prod
        T operator*(const MathVec<T>& vec2) const{
                T result {}; //use default constructor for zero
                for (size_t i=0; i<vec.size(); ++i){
                        result += vec[i]*vec2[i];
                }
                return result;
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

    //--DEFINE OVERLEADED OPS
        // mat-index: can modify
        T& operator()(size_t i, size_t j) {
                return mat[i][j];
        }

        // mat-index: read-only
        const T& operator()(size_t i, size_t j) const {
                return mat[i][j];
        }

        // mat-addition
        MathMat<T> operator+(const MathMat<T>& mat2) const {

                MathMat<T> result(mat.size(), mat[0].size());

                for (size_t i = 0; i < mat.size(); ++i){
                        for (size_t j = 0; j < mat[0].size(); ++j){
                                result(i,j) = mat[i][j] + mat2(i,j);
                        }
                }

                return result;
        }
	// mat-mult
	MathMat<T> operator@(const MathMat<T>& mat2) const {

        	MathMat<T> result(mat.size(), mat2.mat[0].size());

        	for (size_t i = 0; i < mat.size(); ++i){
                	for (size_t j = 0; j < mat2.mat[0].size(); ++j){

                        	result(i,j) = T{};

                        	for (size_t k = 0; k < mat[0].size(); ++k){
                                	result(i,j) += mat[i][k] * mat2(k,j);
                        	}
                	}
        	}

        	return result;
	}	
};
