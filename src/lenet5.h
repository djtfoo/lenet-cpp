#ifndef LENET_5_H
#define LENET_5_H

#include <vector>
#include "map.h"
#include "imagemap.h"
#include "kernel.h"
#include "fcparams.h"

class Lenet5 {
private:
    // variables
    const int IN_LEN = 32;  // (28x28 with padding)
    const int C1_LEN = 28;
    const int S2_LEN = 14;
    const int C3_LEN = 10;
    const int S4_LEN = 5;
    const int C5_LEN = 1;
    const int F6_LEN = 84;
    const int OUT_LEN = 10;

    const int C1_MAPS = 6;
    const int C3_MAPS = 16;
    const int C5_MAPS = 120;

    const int CONV = 5;

    // layer C1
    std::vector<FeatureMap> C1_maps;    // 6 feature maps
    std::vector<Kernel> C1_kernels; // convolution kernel for each feature map
    // layer S2
    std::vector<FeatureMap> S2_maps;    // 6 feature maps
    // layer C3
    std::vector<FeatureMap> C3_maps;    // 16 feature maps
    std::vector<std::vector<Kernel>> C3_kernels;    // 3d convolution kernel for each output feature map
    // layer S4
    std::vector<FeatureMap> S4_maps;    // 16 feature maps
    // layer C5
    std::vector<FeatureMap> C5_maps;    // 120 feature maps
    std::vector<std::vector<Kernel>> C5_kernels;    // 3d convolution kernel for each output feature map
    // layer F6
    std::vector<FCParams> F6_params;    // weights and bias
    std::vector<float> F6_outputs;  // fully-connected layer with 84 outputs
    // OUTPUT layer
    std::vector<FCParams> OUT_params;   // weights and bias
    std::vector<float> OUT_outputs; // fully-connected layer with 10 outputs

    void init();

    // load parameters
    static bool load_weights(Kernel* kernel, int length, const char* filename);
    static bool load_weights(FCParams* params, int length, const char* filename);

    // layer operations
    static void max_pooling_layer(std::vector<FeatureMap>& in, std::vector<FeatureMap>& out, int OUT_LENGTH);
    static void convolution_3d(std::vector<FeatureMap>& in, std::vector<FeatureMap>& out, std::vector<std::vector<Kernel>>& kernels,
        int numKernels, int mapIds[], int n_start, int n_end, int CONV_LENGTH, int LAYER_LENGTH);

    // operations
    static float relu(float in);
    static float convolution(const ImageMap& inputMap, int i_start, int j_start, int convLength, const Kernel& weights);
    static float convolution(const FeatureMap& inputMap, int i_start, int j_start, int convLength, const Kernel& weights);
    static float max_pool(FeatureMap* inputMap, int i_start, int j_start, int poolSize);
    static float fully_connected_output(std::vector<FeatureMap>& inputMaps, const FCParams& params);
    static float fully_connected_output(std::vector<float>& input, const FCParams& params);

public:
    Lenet5() : C1_maps(C1_MAPS), C1_kernels(C1_MAPS), S2_maps(C1_MAPS),
        C3_maps(C3_MAPS), C3_kernels(C3_MAPS), S4_maps(C3_MAPS),
        C5_maps(C5_MAPS), C5_kernels(C5_MAPS),
        F6_params(F6_LEN), F6_outputs(F6_LEN),
        OUT_params(OUT_LEN), OUT_outputs(OUT_LEN)
    {
        // initialize C3 Kernel vectors
        for (int i = 0; i <= 5; ++i) {
            for (int n = 0; n < 3; ++n)
                C3_kernels[i].push_back(Kernel());
        }
        for (int i = 6; i <= 14; ++i) {
            for (int n = 0; n < 4; ++n)
                C3_kernels[i].push_back(Kernel());
        }
        for (int n = 0; n < 6; ++n)
            C3_kernels[15].push_back(Kernel());

        // initalize C5 Kernel vectors
        for (int i = 0; i < C5_MAPS; ++i) { // 120
            for (int n = 0; n < C3_MAPS; ++n)   // 16
                C5_kernels[i].push_back(Kernel());
        }

        init();
    }

    int run_inference(ImageMap* image);
};

#endif