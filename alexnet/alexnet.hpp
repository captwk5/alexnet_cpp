/**
 * 
 * Created by woonuks on 2020-02-11
 * 
 * **/

#ifndef ALEXNET_HPP
#define ALEXNET_HPP

#include "../cnn.hpp"

class AlexNet : public CNN{
public:
    AlexNet(){
        cout << "Hi, I'm AlexNet!" << endl;
    }
    ~AlexNet(){
        cout << "Bye!" << endl;
    }
    void create_network();
    void set_trainig(double learning_rate, string optimizer, string weight_path);
    void training_epoch(int epoch);
    void save_weight_binary(string binary_path, double**** weight, int output_channel, int input_channel, int filter_size);
    void save_weight_binary(string binary_path, double** weight, int length, int next_length);
    void save_weight_binary(string binary_path, double* bias, int length);
    void detroy_network();
protected:
    double training_batch(int step);

private:
    double LR = 0;
    
    double**** input_layer;
    double**** first_layer;
    double**** first_pooling;
    double**** second_layer;
    double**** second_pooling;
    double**** third_layer;
    double**** fourth_layer;
    double**** fifth_layer;
    double**** third_pooling;

    double**** delta_first_layer;
    double**** delta_first_pooling;
    double**** delta_second_layer;
    double**** delta_second_pooling;
    double**** delta_third_layer;
    double**** delta_fourth_layer;
    double**** delta_fifth_layer;

    double** first_fc_layer;
    double** second_fc_layer;
    double** third_fc_layer;
    double** output_layer;

    double**** first_filter;
    double* first_bias;
    double**** second_filter;
    double* second_bias;
    double**** third_filter;
    double* third_bias;
    double**** fourth_filter;
    double* fourth_bias;
    double**** fifth_filter;
    double* fifth_bias;
    double** first_fc_filter;
    double* first_fc_bias;
    double** second_fc_filter;
    double* second_fc_bias;
    double** third_fc_filter;
    double* third_fc_bias;

    double**** first_filter_gradient;
    double**** second_filter_gradient;
    double**** third_filter_gradient;
    double**** fourth_filter_gradient;
    double**** fifth_filter_gradient;

    double** first_fc_gradient;
    double** second_fc_gradient;
    double** third_fc_gradient;

    double* first_bias_gradient;
    double* second_bias_gradient;;
    double* third_bias_gradient;;
    double* fourth_bias_gradient;;
    double* fifth_bias_gradient;;

    double* first_fc_bias_gradient;
    double* second_fc_bias_gradient;
    double* third_fc_bias_gradient;
};

#endif