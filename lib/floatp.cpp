#include <algorithm>
#include "floatp.h"

floatp::floatp() : value(0)
{

}

floatp::floatp(const float& that) : value(that)
{

}

floatp::floatp(float&& that)
{
    std::swap(value, that);
}

floatp::floatp(const floatp& that) : value(that.value)
{

}

floatp::floatp(floatp&& that)
{
    std::swap(value, that.value);
}

floatp::operator float() const
{
    return value;
}

floatp& floatp::operator=(const float& that)
{
    value = that;

    return *this;
}

floatp& floatp::operator=(const floatp& that)
{
    value = that.value;

    return *this;
}

floatp& floatp::operator+=(const float& that)
{
    value += that;

    return *this;
}

floatp& floatp::operator-=(const float& that)
{
    value -= that;

    return *this;
}

floatp& floatp::operator*=(const float& that)
{
    value *= that;

    return *this;
}

floatp& floatp::operator/=(const float& that)
{
    value /= that;

    return *this;
}
