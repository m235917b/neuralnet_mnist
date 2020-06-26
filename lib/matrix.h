#include <ostream>

//for comments/documentary see matrix.cpp / documentary.txt

template<class T> class matrix;
template<class T> matrix<T> operator*(T that1, const matrix<T>& that2);
template<class T> matrix<T> operator/(T that1, const matrix<T>& that2);
template<class T> std::ostream& operator<<(std::ostream& out, const matrix<T>& m);

template<class T> class matrix
{
public:
    matrix();
    matrix(int height, int width);
    matrix(int height, int width, int param);
    matrix(const matrix<T>& owner);
    matrix(matrix<T>&& that) noexcept;
    matrix<T>& operator=(const matrix<T>& that);
    matrix<T>& operator=(matrix<T>&& that);
    void operator=(const std::initializer_list<T>& l);
    friend matrix<T> operator* <>(T that1, const matrix<T>& that2);
    friend matrix<T> operator/ <>(T that1, const matrix<T>& that2);
    friend std::ostream& operator<< <>(std::ostream& out, const matrix<T>& m);
    matrix<T> operator+(const matrix<T>& that) const;
    matrix<T> operator-(const matrix<T>& that) const;
    matrix<T> operator*(const matrix<T>& that) const;
    matrix<T> operator%(const matrix<T>& that) const;
    matrix<T> operator*(T that) const;
    matrix<T> operator/(T that) const;
    void operator+=(const matrix<T>& that);
    void operator-=(const matrix<T>& that);
    void operator*=(const matrix<T>& that);
    void operator%=(const matrix<T>& that);
    void operator*=(T that);
    void operator/=(T that);
    const T& operator()(int x, int y) const;
    const T& operator[](int pos) const;
    T& operator()(int x, int y);
    T& operator[](int pos);
    ~matrix();

    T get(int pos_x, int pos_y) const;
    void set(int pos_x, int pos_y, T value);
    int width() const;
    int height() const;

    void set_zero();
    void randomize();

    matrix<T> column(int collumn) const;
    matrix<T> row(int row) const;
    matrix<T> column_transpose(int c) const;
    matrix<T> row_transpose(int r) const;
    void set_column(int collumn, const matrix<T> &mat);
    void set_row(int row, const matrix<T> &mat);

    float abs() const;
    float sum() const;
    matrix<T> transpose() const;

    static matrix<T> identity(int size);
    static matrix<T> random(int height, int width);

private:
    int m_width, m_height;
    T *mat_ptr;

    T get_lin(int pos);
    void set_lin(int pos, T value);
};

//multiplies each entry by a value
template<class T> matrix<T> operator*(T that1, const matrix<T> that2)
{
    return that2 * that1;
}

//divides each entry by a value
template<class T> matrix<T> operator/(T that1, const matrix<T> that2)
{
    return that2 / that1;
}

template<class T> std::ostream& operator<<(std::ostream& out, const matrix<T>& m)
{
    //run through every row
    for(int x = 0; x < m.m_height; ++x)
    {
        //run through every column
        for(int y = 0; y < m.m_width; ++y)
        {
            //get length of entry and fill rest with spaces
            for(int i = 0; i < 14 - std::to_string(m.mat_ptr[x * m.m_width + y]).length(); ++i)
                out << " ";

            //entry at row x and column y
            out << m.mat_ptr[x * m.m_width + y];
        }

        //new line; one row is written
        out << "\n";
    }

    return out;
}
