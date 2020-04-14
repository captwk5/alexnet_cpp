/**
 * 
 * Created by woonuks on 2020-02-11
 * 
 * **/

#ifndef CNN_HPP
#define CNN_HPP

#include <iostream>
#include <cstdlib>
#include <vector>
#include <random>
#include <fstream>
#include <ctime>
#include <cstdio>
#include <thread>
#include <cmath>

#include "opencv2/opencv.hpp"

#include "json.hpp"

using namespace std;
using namespace cv;

struct InputDataInfo{
    int image_class;
    int image_id;
    string image_path;
};

class CNN{
public:
    void set_data(string data_path, int batch_size, bool suffle);

protected:
    int batch_size_;
    int image_size_;
    vector<InputDataInfo> data_meta_;

    static double activation(double input, string activation);
    void convolution(double**** layer, double**** next_layer, double**** weight, int stride, int input_channel, int output_channel, int filter_size, int size, string activation);
    vector<string> max_pooling(double**** layer, double**** pooling_layer, int output_channel, int layer_size, int filter_size, int stride);
    void flatten(double**** layer, double** fc, int layer_size, int channel);
    void fully_connected(double** layer, double** next_layer, double** weight, int next_length, int length, string activation);

    void backpropagation(double**** gradient, double**** layer, double**** delta, int output_channel, int input_channel, int size, int stride, int filter_size);
    void get_local_gradient(double**** local_gradient, double**** weight, double**** previous_local_gradient, double**** previous_layer, int input_channel, int output_channel, int stride, int filter_size, int size);
    void pooling_to_layer(double**** delta_pooling, double**** delta_layer, double**** layer, vector<string> coord_info);

    void weight_update(double**** weight, double**** gradient, int output_channel, int input_channel, int filter_size, double lr);
    void weight_update(double** weight, double** gradient, int length, int next_length, double lr);

    void local_gradient_zero(double**** local_gradient, int channel_size, int size);

    void layer_memory_assign(double**** layer, int batch_size, int channels, int size);
    void layer_memory_assign(char**** layer, int batch_size, int channels, int size);
    void layer_memory_assign(double*** layer, int batch_size, int length, int next_length);
    void layer_memory_assign(double** layer, int batch_size, int length);
    void layer_memory_release(double**** layer, int batch_size, int channel, int size);
    void layer_memory_release(char**** layer, int batch_size, int channel, int size);
    void layer_memory_release(double*** layer, int batch_size, int length);
    void layer_memory_release(double** layer, int batch_size);

    vector<string> split(string str, char delimiter);

    void set_weight(double**** weight, string weight_path);
    void set_weight(double** weight, string weight_path);

    void transpose_filter(double** weight, double** transpose_weight, int kernel_size);
private:

};

#endif
