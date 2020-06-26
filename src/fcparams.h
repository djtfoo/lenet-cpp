#ifndef FCPARAMS_H
#define FCPARAMS_H

#include <string>
#include <sstream>

class FCParams {
private:
    int _length;
    float* _weights;
    float _bias;

public:
    FCParams() : _length(0), _weights(nullptr), _bias(0.f) {}

    FCParams(int len) : _length(len), _weights(nullptr), _bias(0.f) {
        _weights = new float[len];
    }
    ~FCParams() {
        if (_weights != nullptr) {
            delete[] _weights;
        }
    }
    
    void init(int len) {

        if (_weights != nullptr) {
            delete[] _weights;
            printf("OMG\n");
        }

        _length = len;
        _weights = new float[len];
    }

    void set_weight(float val, int i) {
        _weights[i] = val;
    }

    void set_bias(float bias) {
        _bias = bias;
    }

    std::string to_string() {

        std::ostringstream ss;
        // weights
        for (int i = 0; i < _length; ++i) {
            ss << _weights[i];
            if (i < _length - 1)
                ss << " ";
        }
        
        // bias (next line)
        ss << "\n";
        ss << _bias;

        return ss.str();
    }

    friend class Lenet5;
};

#endif