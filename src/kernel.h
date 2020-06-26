#ifndef KERNEL_H
#define KERNEL_H

#include <string>
#include <sstream>
#include "map.h"

class Kernel : public Map<float> {
private:
    float _bias;    // bias value

public:
    // default constructor for dynamic allocation
    Kernel() : Map(), _bias(0.f) {}

    Kernel(int len) : Map(len), _bias(0.f) {}
    virtual ~Kernel() {}

    void set_bias(float bias) {
        _bias = bias;
    }

    std::string to_string() {

        std::ostringstream ss;
        ss << Map::to_string(); // weights
        ss << "\n";
        ss << _bias;   // bias

        return ss.str();
    }

    friend class Lenet5;
};

#endif