#include <sstream>
#include <random>
#include <math.h>
#include "matrix.h"

/*
This class adds 2-dimensional matrices as a type and offers many functions for them.
It is optimized for class "network" and therefore contains some custom functions for it, that can be used seprately though.
This class uses templates to be able to create matrices of different data types; every matrix is stored 1-dimensional,
so position of entries must be calculated.
*/

//standard constructor
template<class T> matrix<T>::matrix() : m_height(0), m_width(0), mat_ptr(nullptr)
{

}

//custom constructor
template<class T> matrix<T>::matrix(int m_height, int m_width) : m_height(m_height), m_width(m_width),
    mat_ptr(new T[m_height * m_width])
{

}

template<class T> matrix<T>::matrix(int m_height, int m_width, int param) : m_height(m_height), m_width(m_width),
    mat_ptr(new T[m_height * m_width])
{

}

//copy constructor; needed, because otherwise copies of an instance of "matrix" would only contain a copy of the pointer mat_ptr an thus pointing to the same adress as the original object
template<class T> matrix<T>::matrix(const matrix<T>& owner) : m_height(owner.m_height), m_width(owner.m_width),
    mat_ptr(new T[owner.m_height * owner.m_width])
{
    //copy entrys from original mat_ptr to the new allocated memory space for this copy
    std::copy(owner.mat_ptr, owner.mat_ptr + m_height * m_width, mat_ptr);
}

//move constructor
template<class T> matrix<T>::matrix(matrix<T>&& that) noexcept : matrix<T>()
{
    std::swap(m_height, that.m_height);
    std::swap(m_width, that.m_width);
    std::swap(mat_ptr, that.mat_ptr);
}

//copy assignment operator; same as copy constructor but for copies created by assignments (e.g. matrix1 = matrix2)
template<class T> matrix<T>& matrix<T>::operator=(const matrix<T>& that)
{
    if(this != &that)
    {
        m_height = that.m_height;
        m_width = that.m_width;

        delete[] mat_ptr;

        mat_ptr = new T[m_width*m_height];

        std::copy(that.mat_ptr, that.mat_ptr + m_height * m_width, mat_ptr);
    }

    return *this;
}

//copy move operator
template<class T> matrix<T>& matrix<T>::operator=(matrix<T>&& that)
{
    std::swap(m_height, that.m_height);
    std::swap(m_width, that.m_width);
    std::swap(mat_ptr, that.mat_ptr);

    return *this;
}

//assignment operator for initializer list
template<class T> void matrix<T>::operator=(const std::initializer_list<T>& l)
{
    if(l.size() > m_width * m_height)
    {
        return;
    }

    std::copy(l.begin(), l.end(), mat_ptr);
}

//adding two matrices
template<class T> matrix<T> matrix<T>::operator+(const matrix<T>& that) const
{
    //create matrix for result
    matrix<T> out(m_height, m_width);

    //check dimensions
    if(m_height != that.m_height || m_width != that.m_width)
    {
        return out;
    }

    T* m_ptr_o = out.mat_ptr;
    T* m_ptr = that.mat_ptr;
    //add entrys of matching position and save result to "out"
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[i] + m_ptr[i];
    }

    //return result
    return out;
}

//subtracting two matrices
template<class T> matrix<T> matrix<T>::operator-(const matrix<T>& that) const
{
    //create matrix for result
    matrix<T> out(m_height, m_width);

    //check dimensions
    if(m_height != that.m_height || m_width != that.m_width)
    {
        return out;
    }

    T* m_ptr_o = out.mat_ptr;
    T* m_ptr = that.mat_ptr;
    //subtract entrys of matching position and save result to "out"
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[i] - m_ptr[i];
    }

    //return result
    return out;
}

//multiply two matrices
template<class T> matrix<T> matrix<T>::operator*(const matrix<T>& that) const
{
    int t_w = that.m_width;
    //create matrix for result
    matrix<T> out(m_height, t_w);

    //check dimensions
    if(m_width != that.m_height)
    {
        return out;
    }

    matrix<T> transp = that.transpose();

    T* m_ptr = transp.mat_ptr;
    T* m_ptr_o = out.mat_ptr;
    int h = transp.m_height, w = transp.m_width;

    #pragma omp parallel for
    for(int x = m_height - 1; x >= 0; --x)
    {
        for(int y = h - 1; y >= 0; --y)
        {
            m_ptr_o[x * t_w + y] = mat_ptr[x * m_width] * m_ptr[y * w];
            for(int i = w - 1; i > 0; --i)
            {
                m_ptr_o[x * t_w + y] += mat_ptr[x * m_width + i] * m_ptr[y * w + i];
            }
        }
    }

    //return result
    return out;
}

//multiplies two matrices entrywise and returns the resulting matrix
template<class T> matrix<T> matrix<T>::operator%(const matrix<T>& that) const
{
    //create matrix for result
    matrix<T> out(m_height, m_width);

    //check dimensions
    if(m_height != that.m_height || m_width != that.m_width)
    {
        return out;
    }

    T* m_ptr_o = out.mat_ptr;
    T* m_ptr = that.mat_ptr;
    //multiply entrys of matching position and save result to "out"
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[i] * m_ptr[i];
    }

    //return result
    return out;
}

//multiplies each entry by a value
template<class T> matrix<T> matrix<T>::operator*(T that) const
{
    //create matrix for result
    matrix<T> out(m_height, m_width);

    T* m_ptr_o = out.mat_ptr;
    //multiply each entry by value of "that"
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[i] * that;
    }

    //return result
    return out;
}

//divides each entry by a value
template<class T> matrix<T> matrix<T>::operator/(T that) const
{
    //create matrix for result
    matrix<T> out(m_height, m_width);

    T* m_ptr_o = out.mat_ptr;
    //divide each entry by value of "that"
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[i] / that;
    }

    //return result
    return out;
}

//add matrix to this one
template<class T> void matrix<T>::operator+=(const matrix<T>& that)
{
    //check dimensions
    if(m_height != that.m_height || m_width != that.m_width)
    {
        return;
    }

    T* m_ptr = that.mat_ptr;
    //add entrys of matching position
    #pragma omp parallel for
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        mat_ptr[i] += m_ptr[i];
    }
}

//subtract a matrix from this one
template<class T> void matrix<T>::operator-=(const matrix<T>& that)
{
    //check dimensions
    if(m_height != that.m_height || m_width != that.m_width)
    {
        return;
    }

    T* m_ptr = that.mat_ptr;
    //subtract entrys of matching position
    #pragma omp parallel for
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        mat_ptr[i] -= m_ptr[i];
    }
}

//multiply matrix with this one
template<class T> void matrix<T>::operator*=(const matrix<T>& that)
{
    int t_w = that.m_width;
    //create matrix for result
    matrix<T> out(m_height, t_w);

    //check dimensions
    if(m_width != that.m_height)
    {
        return;
    }

    matrix<T> transp = that.transpose();

    T *m_ptr = transp.mat_ptr;
    T* m_ptr_o = out.mat_ptr;
    int h = transp.m_height, w = transp.m_width;

    #pragma omp parallel for
    for(int x = m_height - 1; x >= 0; --x)
    {
        for(int y = h - 1; y >= 0; --y)
        {
            for(int i = w - 1; i >= 0; --i)
            {
                m_ptr_o[x * t_w + y] += mat_ptr[x * m_width + i] * m_ptr[y * w + i];
            }
        }
    }

    //set this to resulting matrix
    *this = out;
}

//multiplies this matrix with another
template<class T> void matrix<T>::operator%=(const matrix<T>& that)
{
    //check dimensions
    if(m_height != that.m_height || m_width != that.m_width)
    {
        return;
    }

    T* m_ptr = that.mat_ptr;
    //multiply entrys of matching position
    for(int i = m_height * m_width - 1; i >= 0; --i)
    {
        mat_ptr[i] *= m_ptr[i];
    }
}

//multiplies each entry by a value
template<class T> void matrix<T>::operator*=(const T that)
{
    //multiplyeach entry by value of "that"
    for(int i = m_height * m_width - 1; i >= 0; --i)
    {
        mat_ptr[i] *= that;
    }
}

//divides each entry by a value
template<class T> void matrix<T>::operator/=(const T that)
{
    //divide each entry by value of "that"
    for(int i = m_height * m_width - 1; i >= 0; --i)
    {
        mat_ptr[i] /= that;
    }
}

/*
Returns position (x, y). A matrix m can then be accessed
by m(x, y) and the value at (x, y) can be changed by m(x, y) = value.
*/
template<class T> const T& matrix<T>::operator()(int x, int y) const
{
    //calculate position and return reference to entry
    return mat_ptr[x * m_width + y];
}

//get position x in the corresponding 1-dim matrix; similar to operator ()
template<class T> const T& matrix<T>::operator[](int pos) const
{
    //return reference to entry
    return mat_ptr[pos];
}

template<class T> T& matrix<T>::operator()(int x, int y)
{
    //calculate position and return reference to entry
    return mat_ptr[x * m_width + y];
}

template<class T> T& matrix<T>::operator[](int pos)
{
    //return reference to entry
    return mat_ptr[pos];
}

//destructor
template<class T> matrix<T>::~matrix()
{
    delete[] mat_ptr;
}

//returns entry in row pos_x and column pos_y
template<class T> T matrix<T>::get(int pos_x, int pos_y) const
{
    //check if positions are valid
    if(pos_x >= m_height || pos_y >= m_width || pos_x < 0 || pos_y < 0)
    {
        return 0;
    }

    //calculate position and return entry
    return mat_ptr[pos_x * m_width + pos_y];
}

//sets entry in row pos_x and column pos_y to "value"
template<class T> void matrix<T>::set(int pos_x, int pos_y, T value)
{
    if(pos_x >= m_height || pos_y >= m_width || pos_x < 0 || pos_y < 0)
    {
        return;
    }

    //calculate position and set entry
    mat_ptr[pos_x * m_width + pos_y] = value;
}

//returns value at 1-dimensional position "pos"; needed for arithmetic methods below
template<class T> T matrix<T>::get_lin(int pos)
{
    //check if positions are valid
    if(pos >= m_width*m_height)
    {
        return 0;
    }

    //return entry
    return mat_ptr[pos];
}

//sets value at 1-dimensional position "pos" to "value"; needed for arithmetic methods below
template<class T> void matrix<T>::set_lin(int pos, T value)
{
    //check if positions are valid
    if(pos >= m_width * m_height)
    {
        return;
    }

    //set entry
    mat_ptr[pos] = value;
}

//return width of matrix
template<class T> int matrix<T>::width() const
{
    return m_width;
}

//return height of matrix
template<class T> int matrix<T>::height() const
{
    return m_height;
}

//set every entry to 0
template<class T> void matrix<T>::set_zero()
{
    for(int i = m_height * m_width - 1; i >= 0 ; --i)
    {
        mat_ptr[i] = 0;
    }
}

//set every entry to a random number between -1 and 1
template<class T> void matrix<T>::randomize(double dist)
{
    std::random_device mch;
    std::default_random_engine generator(mch());
    std::normal_distribution<double> distribution(0.0, dist);

    for(int i = m_height * m_width - 1; i >= 0 ; --i)
    {
        mat_ptr[i] = distribution(generator);
    }
}

//rerturns column at position "column" as a vertical 1-dimensinal matrix
template<class T> matrix<T> matrix<T>::column(int col) const
{
    //create matrix with dimensions of the column
    matrix<T> out(m_height, 1);

    //check if position is valid
    if(col >= m_width || col < 0)
    {
        return out;
    }

    T* m_ptr_o = out.mat_ptr;
    //save values of the column into "out"
    for(int i = m_height - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[i * m_width + col];
    }

    //return copy of column as object
    return out;
}

//rerturns row at position "row" as a horizontal 1-dimensinal matrix
template<class T> matrix<T> matrix<T>::row(int r) const
{
    //create matrix with dimensions of the row
    matrix<T> out(1, m_width);

    //check if position is valid
    if(r >= m_height || r < 0)
    {
        return out;
    }

    T* m_ptr_o = out.mat_ptr;
    //save values of the row into "out"
    for(int i = m_width - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[r * m_width + i];
    }

    //return copy of row as object
    return out;
}

//rerturns row at position r as a vertical 1-dimensinal matrix; see row)
template<class T> matrix<T> matrix<T>::row_transpose(int r) const
{
    matrix<T> out(m_width, 1);

    if(r >= m_height || r < 0)
    {
        return out;
    }

    T* m_ptr_o = out.mat_ptr;
    for(int i = m_width - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[r * m_width + i];
    }

    return out;
}

//rerturns column at position c as a horizontal 1-dimensinal matrix; see column()
template<class T> matrix<T> matrix<T>::column_transpose(int c) const
{
    matrix<T> out(1, m_height);

    if(c >= m_width || c < 0)
    {
        return out;
    }

    T* m_ptr_o = out.mat_ptr;
    for(int i = m_height - 1; i >= 0; --i)
    {
        m_ptr_o[i] = mat_ptr[i * m_width + c];
    }

    return out;
}

//set values of column in position "column" to those of "mat"; "mat" must be 1-dimensional vertical matrix with same height as this matrix
template<class T> void matrix<T>::set_column(int col, const matrix<T>& mat)
{
    //check if position is valid
    if(col > m_width || col < 0 || mat.m_height != m_height || mat.m_width != 1)
    {
        return;
    }

    T* m_ptr = mat.mat_ptr;
    //set values of the column to the ones in "mat"
    for(int i = m_height - 1; i >= 0; --i)
    {
        mat_ptr[i * m_width + col] = m_ptr[i];
    }
}

//set values of row in position "row" to those of "mat"; "mat" must be 1-dimensional horizontal matrix with same width as this matrix
template<class T> void matrix<T>::set_row(int r, const matrix<T>& mat)
{
    //check if position is valid
    if(r > m_height || r < 0 || mat.m_height != 1 || mat.m_width != m_width)
    {
        return;
    }

    T* m_ptr = mat.mat_ptr;
    //set values of the row to the ones in "mat"
    for(int i = m_width - 1; i >= 0; --i)
    {
        mat_ptr[r * m_width + i] = m_ptr[i];
    }
}

//returns absolute value of matrix
template<class T> float matrix<T>::abs() const
{
    //variable to save sum
    float out = 0;

    //add every entry squared
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        out += mat_ptr[i] * mat_ptr[i];
    }

    //return square root of sum
    return sqrt(out);
}

//returns sum of every entry
template<class T> float matrix<T>::sum() const
{
    //variable to save sum
    float out = 0;

    //add every entry
    for(int i = m_width * m_height - 1; i >= 0; --i)
    {
        out += mat_ptr[i];
    }

    //return sum
    return out;
}

//transpose matrix
template<class T> matrix<T> matrix<T>::transpose() const
{
    //create matrix with new height = width and new width = height
    matrix<T> out = matrix<T>(m_width, m_height);

    T* m_ptr_o = out.mat_ptr;
    //switch rows and columns
    for(int x = m_height - 1; x >= 0; --x)
    {
        for(int y = m_width - 1; y >= 0; --y)
        {
            m_ptr_o[y * m_height + x] = mat_ptr[x * m_width + y];
        }
    }

    return out;
}

//returns the identity matrix
template<class T> matrix<T> matrix<T>::identity(int size)
{
    matrix<T> id = matrix<T>(size, size);

    T* m_ptr = id.mat_ptr;
    for(int i = size - 1; i >= 0; --i)
    {
        m_ptr[i * size + i] = 1;
    }

    return id;
}

//returns a matrix with random numbers between -1 and 1
template<class T> matrix<T> matrix<T>::random(int height, int width, double dist)
{
    matrix<T> out = matrix<T>(height, width);

    std::random_device mch;
    std::default_random_engine generator(mch());
    std::normal_distribution<double> distribution(0.0, dist);

    T* m_ptr_o = out.mat_ptr;
    for(int i = height * width - 1; i >= 0 ; --i)
    {
        m_ptr_o[i] = distribution(generator);
    }

    return out;
}

template<class T> matrix<T> matrix<T>::null(int height, int width)
{
    matrix<T> out = matrix<T>(height, width);
    T* m_ptr_o = out.mat_ptr;
    //fill array with 0s
    for(int i = height * width - 1; i >= 0; --i)
    {
        m_ptr_o[i] = 0;
    }

    return out;
}

//set valid template types
template class matrix<int>;
template class matrix<float>;
template class matrix<double>;
template class matrix<short>;
template class matrix<long>;
