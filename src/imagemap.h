#ifndef IMAGE_MAP_H
#define IMAGE_MAP_H

#include "map.h"

class ImageMap : public Map<char> {
private:
    char _label;

public:
    ImageMap(int length) : Map(length), _label(NULL) {}
    ImageMap(int length, char label) : Map(length), _label(label) {}

    virtual ~ImageMap() {}

    char get_label() { return _label; }

    void set_label(char label) {
        _label = label;
    }
};

#endif