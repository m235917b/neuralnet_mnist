//float type with additional parameter
struct floatp
{
    float value;
    int param = 0;

    floatp();
    floatp(const float& that);
    floatp(float&& that);
    floatp(const floatp& that);
    floatp(floatp&& that);

    operator float() const;
    floatp& operator=(const float& that);
    floatp& operator=(const floatp& that);
    floatp& operator+=(const float& that);
    floatp& operator-=(const float& that);
    floatp& operator*=(const float& that);
    floatp& operator/=(const float& that);

};
