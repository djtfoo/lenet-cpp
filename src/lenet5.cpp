#include <fstream>
#include "lenet5.h"

#define MAXCHAR 1000

void Lenet5::init() {

    // initialize C1 maps and kernels
    for (int n = 0; n < C1_maps.size(); ++n) {

        // initialize feature map
        C1_maps[n].init(C1_LEN);

        // initialize kernel
        C1_kernels[n].init(CONV);
        // load parameters
        char c1_kernel_file[50];
        sprintf_s(c1_kernel_file, "params/kernel_c1_m%d.txt", n);
        load_weights(&(C1_kernels[n]), CONV, c1_kernel_file);
    }

    // initialize S2 maps
    for (int n = 0; n < S2_maps.size(); ++n) {
        // initialize feature map
        S2_maps[n].init(S2_LEN);
    }


    // initialize C3 maps
    for (int n = 0; n < C3_maps.size(); ++n) {
        // initialize feature map
        C3_maps[n].init(C3_LEN);
    }

    // initialize C3 kernels
    // 6 maps 3rd dimension = 3
    for (int n = 0; n <= 5; ++n) {
        for (int k = 0; k < 3; ++k) {  // 1 kernel for each 3rd dimension of convolution
            C3_kernels[n][k].init(CONV);
            // load parameters
            char c3_kernel_file[50];
            sprintf_s(c3_kernel_file, "params/kernel_c3_m%d_%d.txt", n, k);
            load_weights(&(C3_kernels[n][k]), CONV, c3_kernel_file);
        }
    }
    // 9 maps 3rd dimension = 4
    for (int n = 6; n <= 14; ++n) {
        for (int k = 0; k < 4; ++k) {  // 1 kernel for each 3rd dimension of convolution
            C3_kernels[n][k].init(CONV);
            // load parameters
            char c3_kernel_file[50];
            sprintf_s(c3_kernel_file, "params/kernel_c3_m%d_%d.txt", n, k);
            load_weights(&(C3_kernels[n][k]), CONV, c3_kernel_file);
        }
    }
    // 1 map 3rd dimension = 6
    for (int k = 0; k < 6; ++k) {  // 1 kernel for each 3rd dimension of convolution
        C3_kernels[15][k].init(CONV);
        // load parameters
        char c3_kernel_file[50];
        sprintf_s(c3_kernel_file, "params/kernel_c3_m%d_%d.txt", 15, k);
        load_weights(&(C3_kernels[15][k]), CONV, c3_kernel_file);
    }


    // initialize S4 maps
    for (int n = 0; n < S4_maps.size(); ++n) {
        // initialize feature map
        S4_maps[n].init(S4_LEN);
    }

    // initialize C5 maps and kernels
    for (int n = 0; n < C5_MAPS; ++n) {

        // initialize feature map
        C5_maps[n].init(C5_LEN);

        // initialize kernels (convolution kernel for each feature map)
        for (int k = 0; k < C3_MAPS; ++k) {  // 1 kernel for each 3rd dimension of convolution
            C5_kernels[n][k].init(CONV);
            // load parameters
            char c5_kernel_file[50];
            sprintf_s(c5_kernel_file, "params/kernel_c5_m%d_%d.txt", n, k);
            load_weights(&(C5_kernels[n][k]), CONV, c5_kernel_file);
        }
    }


    // initialize F6 parameters
    for (int n = 0; n < F6_LEN; ++n) {
        // initialize parameters
        F6_params[n].init(C5_MAPS);

        // read parameters
        char f6_kernel_file[25];
        sprintf_s(f6_kernel_file, "params/fc_f6_out%d.txt", n);
        load_weights(&(F6_params[n]), C5_MAPS, f6_kernel_file);
    }

    // initialize OUTPUT parameters
    for (int n = 0; n < OUT_LEN; ++n) {
        // initialize parameters
        OUT_params[n].init(F6_LEN);

        // read parameters
        char last_kernel_file[30];
        sprintf_s(last_kernel_file, "params/fc_last_out%d.txt", n);
        load_weights(&(OUT_params[n]), F6_LEN, last_kernel_file);
    }
}


void Lenet5::convolution_3d(std::vector<FeatureMap>& in, std::vector<FeatureMap>& out, std::vector<std::vector<Kernel>>& kernels,
    int numKernels, int mapIds[], int n_start, int n_end, int CONV_LENGTH, int LAYER_LENGTH)
{
    // perform convolution
    for (int n = n_start; n <= n_end; ++n) {
        //printf("Convolution: Map %d\n", n);
        for (int i = 0; i < LAYER_LENGTH; ++i) {  // stride = 1
            for (int j = 0; j < LAYER_LENGTH; ++j) {  // stride = 1
                float convOut = 0;
                // 3-dimensional convolution
                for (int k = 0; k < numKernels; ++k) {
                    convOut += convolution(in[mapIds[k]], i, j, CONV_LENGTH, kernels[n][k]);
                }
                convOut = relu(convOut);
                out[n].set_cell(convOut, i, j);
                //printf("%.2f ", convOut);
            }
            //printf("\n");
        }

        // update map indexes
        for (int k = 0; k < numKernels; ++k) {
            mapIds[k] = (mapIds[k] + 1) % 6;
        }
    }
}

void Lenet5::max_pooling_layer(std::vector<FeatureMap>& in, std::vector<FeatureMap>& out, int OUT_LENGTH) {

    // perform max pooling
    for (int n = 0; n < out.size(); ++n) {
        //printf("Pooling: Map %d\n", n);
        for (int i = 0; i < OUT_LENGTH; ++i) {
            for (int j = 0; j < OUT_LENGTH; ++j) {
                float max = max_pool(&(in[n]), i * 2, j * 2, 2); // stride = 2
                out[n].set_cell(max, i, j);
                //printf("%.2f ", max);
            }
            //printf("\n");
        }
    }
}

int Lenet5::run_inference(ImageMap* image) {

    //image->print();

    // layer C1 convolution
    for (int n = 0; n < C1_maps.size(); ++n) {

        //printf("Convolution: Map %d\n", n);
        for (int i = 0; i < C1_LEN; ++i) {  // stride = 1
            for (int j = 0; j < C1_LEN; ++j) {  // stride = 1
                float convOut = convolution(*image, i, j, CONV, C1_kernels[n]);
                C1_maps[n].set_cell(relu(convOut), i, j);
                //printf("%.2f ", convOut);
            }
            //printf("\n");
        }
    }

    // layer S2 max pooling
    max_pooling_layer(C1_maps, S2_maps, S2_LEN);

    // layer C3 convolution
    // 1st 6 C3 feature maps (#0 to #5): take inputs from every contiguous subset of 3 feature maps
    int initial_ids_0[] = { 0, 1, 2 };
    convolution_3d(S2_maps, C3_maps, C3_kernels, 3, initial_ids_0, 0, 5, CONV, C3_LEN);
    // next 6 C3 feature maps (#6 to #11): take inputs from every contiguous subset of 4 feature maps
    int initial_ids_1[] = { 0, 1, 2, 3 };
    convolution_3d(S2_maps, C3_maps, C3_kernels, 4, initial_ids_1, 6, 11, CONV, C3_LEN);
    // next 3 C3 feature maps (#12 to #14): take inputs from some discontinous subsets of 4 feature maps
    int initial_ids_2[] = { 0, 1, 3, 4 };
    convolution_3d(S2_maps, C3_maps, C3_kernels, 4, initial_ids_2, 12, 14, CONV, C3_LEN);
    // last 1 C3 feature map (#15): takes input from all 6 S2 feature maps
    int initial_ids_3[] = { 0, 1, 2, 3, 4, 5 };
    convolution_3d(S2_maps, C3_maps, C3_kernels, 6, initial_ids_3, 15, 15, CONV, C3_LEN);


    // layer S4 max pooling
    max_pooling_layer(C3_maps, S4_maps, S4_LEN);

    // layer C5 convolution
    // each feature map takes input from all 16 feature maps
    int c5_map_ids[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };    // hardcoded bc lazy to change the method
    convolution_3d(S4_maps, C5_maps, C5_kernels, 16, c5_map_ids, 0, 119, CONV, C5_LEN);

    // layer F6 fully-connected
    //printf("FC LAYER F6:\n");
    for (int n = 0; n < F6_LEN; ++n) {
        //fully_connected_output + ReLU
        F6_outputs[n] = relu(fully_connected_output(C5_maps, F6_params[n]));
        //printf("%.2f ", F6_outputs[n]);
    }

    // OUTPUT layer: fully-connected (skip softmax function), 10 outputs
    //printf("OUTPUT LAYER:\n");
    for (int n = 0; n < OUT_LEN; ++n) {
        // fully connected
        OUT_outputs[n] = fully_connected_output(F6_outputs, OUT_params[n]);
        //printf("%.2f ", OUT_outputs[n]);
    }

    // treat the largest output as the NN's prediction
    int maxIdx = 0;
    //printf("OUTPUTS:");
    //printf("%.4f ", OUT_outputs[0]);
    for (int i = 1; i < OUT_LEN; ++i) {
        //printf("%.4f ", OUT_outputs[i]);
        if (OUT_outputs[i] >= OUT_outputs[maxIdx])  // take the "later" one; as this is the implementation for Verilog for now
        {
            //printf("New max! ");
            maxIdx = i;
        }
    }
    //printf("\n");

    return maxIdx;
}

float Lenet5::convolution(const ImageMap& inputMap, int i_start, int j_start, int convLength, const Kernel& weights) {

    float convResult = 0;
    for (int i = 0; i < convLength; ++i) {
        for (int j = 0; j < convLength; ++j) {
            convResult += (float)(inputMap._values[i + i_start][j + j_start]) * weights._values[i][j];

            //printf("input:%.2f, weight:%.2f, Conv:%.2f ", (float)(inputMap._values[i + i_start][j + j_start]), weights._values[i][j], convResult);
        }
    }

    return convResult + weights._bias;
}

float Lenet5::convolution(const FeatureMap& inputMap, int i_start, int j_start, int convLength, const Kernel& weights) {

    float convResult = 0;
    for (int i = 0; i < convLength; ++i) {
        for (int j = 0; j < convLength; ++j) {
            convResult += inputMap._values[i + i_start][j + j_start] * weights._values[i][j];
        }
    }

    return convResult + weights._bias;
}

float Lenet5::relu(float in) {
    return (in < 0.f) ? 0.f : in;
}

float Lenet5::max_pool(FeatureMap* inputMap, int i_start, int j_start, int poolSize) {

    int max = inputMap->_values[i_start][j_start];
    for (int i = 0; i < poolSize; i++) {
        for (int j = 0; j < poolSize; j++) {
            float thisVal = inputMap->_values[i + i_start][j + j_start];
            if (thisVal > max)
                max = thisVal;
        }
    }

    return max;
}

float Lenet5::fully_connected_output(std::vector<FeatureMap>& inputMaps, const FCParams& params) {

    float output = 0;
    int length = inputMaps[0]._length;
    for (int n = 0; n < inputMaps.size(); ++n) {    // loop through all input maps
        for (int i = 0; i < length; ++i) {      // 1st-Dimension of map
            for (int j = 0; j < length; ++j) {  // 2nd-Dimension of map
                output += inputMaps[n]._values[i][j] * params._weights[i * length + j];
            }
        }
    }

    return output + params._bias;
}

float Lenet5::fully_connected_output(std::vector<float>& input, const FCParams& params) {

    float output = 0;
    for (int i = 0; i < input.size(); ++i) {
        output += input[i] * params._weights[i];
    }

    return output + params._bias;
}

bool Lenet5::load_weights(FCParams* params, int length, const char* filename) {

    FILE* fp;
    errno_t err;
    char str[MAXCHAR];

    // open file
    if ((err = fopen_s(&fp, filename, "r")) != 0) { // file opened unsuccessfully
        fprintf(stderr, "cannot open file '%s'\n", filename);

        // randomly generate parameters
        for (int i = 0; i < length; ++i) {
            params->set_weight(0.02f * (rand() % 100 - 50), i); // initialize random
        }
        params->set_bias(0.01f * (rand() % 2000 - 1000));   // initialize random

        // export parameters to text file
        std::ofstream write(filename);
        write << params->to_string();
        write.close();

        return false;
    }

    // read file
    int i = 0;
    while (fgets(str, MAXCHAR, fp) != NULL) {
        char* token, * next_token;
        token = strtok_s(str, " ", &next_token);

        // check if is bias
        if (i >= 1) {
            float fVal = std::stof(str);
            params->set_bias(fVal);
            //printf("bias: %.2f", fVal);
        }
        else {
            // loop through the string to extract all other tokens
            int j = 0;
            while (token != NULL) {
                // convert string to float
                float fVal = std::stof(token);
                //printf("%.2f ", fVal); // print each token
                // set respective cell in ImageMap
                params->set_weight(fVal, j);
                token = strtok_s(NULL, " ", &next_token);
                j++;
            }
        }

        //printf("\n");
        i++;
    }
    fclose(fp);

    return true;
}

bool Lenet5::load_weights(Kernel* kernel, int length, const char* filename) {

    FILE* fp;
    errno_t err;
    char str[1600];

    // open file
    if ((err = fopen_s(&fp, filename, "r")) != 0) { // file opened unsuccessfully
        fprintf(stderr, "cannot open file '%s'\n", filename);

        // randomly generate parameters
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                kernel->set_cell(0.02f * (rand() % 100 - 50), i, j);    // initialize random
            }
        }
        kernel->set_bias(0.01f * (rand() % 2000 - 1000));   // initialize random

        // export parameters to text file
        std::ofstream write(filename);
        write << kernel->to_string();
        write.close();

        return false;
    }

    // read file
    int i = 0;
    while (fgets(str, MAXCHAR, fp) != NULL) {
        char* token, * next_token;
        token = strtok_s(str, " ", &next_token);

        // check if is bias
        if (i >= length) {
            float fVal = std::stof(str);
            kernel->set_bias(fVal);
            //printf("bias: %.2f", fVal);
        }
        else {
            // loop through the string to extract all other tokens
            int j = 0;
            while (token != NULL) {
                // convert string to float
                float fVal = std::stof(token);
                //printf("%.2f ", fVal); // print each token
                // set respective cell in ImageMap
                kernel->set_cell(fVal, i, j);
                token = strtok_s(NULL, " ", &next_token);
                j++;
            }
        }

        //printf("\n");
        i++;
    }
    fclose(fp);

    return true;
}