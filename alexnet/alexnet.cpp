//
// Created by wkh on 2020/04/22.
//

#include "alexnet.hpp"

void AlexNet::create_network(){
    /* AlexNet Network Memory Assign */
    input_layer = new double***[batch_size_];
    layer_memory_assign(input_layer, batch_size_, 3, 224);
    first_layer = new double***[batch_size_];
    layer_memory_assign(first_layer, batch_size_, 64, 56);
    first_pooling = new double***[batch_size_];
    layer_memory_assign(first_pooling, batch_size_, 64, 27);
    second_layer = new double***[batch_size_];
    layer_memory_assign(second_layer, batch_size_, 192, 27);
    second_pooling = new double***[batch_size_];
    layer_memory_assign(second_pooling, batch_size_, 192, 13);
    third_layer = new double***[batch_size_];
    layer_memory_assign(third_layer, batch_size_, 384, 13);
    fourth_layer = new double***[batch_size_];
    layer_memory_assign(fourth_layer, batch_size_, 256, 13);
    fifth_layer = new double***[batch_size_];
    layer_memory_assign(fifth_layer, batch_size_, 256, 13);
    third_pooling = new double***[batch_size_];
    layer_memory_assign(third_pooling, batch_size_, 256, 6);

    delta_first_layer = new double***[batch_size_];
    layer_memory_assign(delta_first_layer, batch_size_, 64, 56);
    delta_first_pooling = new double***[batch_size_];
    layer_memory_assign(delta_first_pooling, batch_size_, 64, 27);
    delta_second_layer = new double***[batch_size_];
    layer_memory_assign(delta_second_layer, batch_size_, 192, 27);
    delta_second_pooling = new double***[batch_size_];
    layer_memory_assign(delta_second_pooling, batch_size_, 192, 13);
    delta_third_layer = new double***[batch_size_];
    layer_memory_assign(delta_third_layer, batch_size_, 384, 13);
    delta_fourth_layer = new double***[batch_size_];
    layer_memory_assign(delta_fourth_layer, batch_size_, 256, 13);
    delta_fifth_layer = new double***[batch_size_];
    layer_memory_assign(delta_fifth_layer, batch_size_, 256, 13);

    first_fc_layer = new double*[batch_size_];
    layer_memory_assign(first_fc_layer, batch_size_, 6 * 6 * 256);
    second_fc_layer = new double*[batch_size_];
    layer_memory_assign(second_fc_layer, batch_size_, 4096);
    third_fc_layer = new double*[batch_size_];
    layer_memory_assign(third_fc_layer, batch_size_, 4096);
    output_layer = new double*[batch_size_];
    layer_memory_assign(output_layer, batch_size_, 1000);

    first_filter = new double***[64];
    layer_memory_assign(first_filter, 64, 3, 11);
    second_filter = new double***[192];
    layer_memory_assign(second_filter, 192, 64, 5);
    third_filter = new double***[384];
    layer_memory_assign(third_filter, 384, 192, 3);
    fourth_filter = new double***[256];
    layer_memory_assign(fourth_filter, 256, 384, 3);
    fifth_filter = new double***[256];
    layer_memory_assign(fifth_filter, 256, 256, 3);

    first_fc_filter = new double*[4096];
    layer_memory_assign(first_fc_filter, 4096, 6 * 6 * 256);
    second_fc_filter = new double*[4096];
    layer_memory_assign(second_fc_filter, 4096, 4096);
    third_fc_filter = new double*[1000];
    layer_memory_assign(third_fc_filter, 1000, 4096);

    first_bias = new double[64];
    second_bias = new double[192];
    third_bias = new double[384];
    fourth_bias = new double[256];
    fifth_bias = new double[256];

    first_fc_bias = new double[4096];
    second_fc_bias = new double[4096];
    third_fc_bias = new double[1000];

    first_bias_gradient = new double[64];
    second_bias_gradient = new double[192];
    third_bias_gradient = new double[384];
    fourth_bias_gradient = new double[256];
    fifth_bias_gradient = new double[256];

    first_fc_bias_gradient = new double[4096];
    second_fc_bias_gradient = new double[4096];
    third_fc_bias_gradient = new double[1000];

    first_filter_gradient = new double***[64];
    layer_memory_assign(first_filter_gradient, 64, 3, 11);
    second_filter_gradient = new double***[192];
    layer_memory_assign(second_filter_gradient, 192, 64, 5);
    third_filter_gradient = new double***[384];
    layer_memory_assign(third_filter_gradient, 384, 192, 3);
    fourth_filter_gradient = new double***[256];
    layer_memory_assign(fourth_filter_gradient, 256, 384, 3);
    fifth_filter_gradient = new double***[256];
    layer_memory_assign(fifth_filter_gradient, 256, 256, 3);

    first_fc_gradient = new double*[4096];
    layer_memory_assign(first_fc_gradient, 4096, 6 * 6 * 256);
    second_fc_gradient = new double*[4096];
    layer_memory_assign(second_fc_gradient, 4096, 4096);
    third_fc_gradient = new double*[1000];
    layer_memory_assign(third_fc_gradient, 1000, 4096);

}

void AlexNet::set_trainig(double learning_rate, string optimizer, string weight_path){
    LR = learning_rate;
    const unsigned int HOST_NUM_THREAD = std::thread::hardware_concurrency();

    int start_time = static_cast<int>(time(0));

    set_weight(first_filter, weight_path + "64-3-11-11.bin");
    set_weight(second_filter, weight_path +"192-64-5-5.bin");
    set_weight(third_filter, weight_path + "384-192-3-3.bin");
    set_weight(fourth_filter, weight_path + "256-384-3-3.bin");
    set_weight(fifth_filter, weight_path + "256-256-3-3.bin");

    set_weight(first_fc_filter, weight_path + "4096-9216.bin");
    set_weight(second_fc_filter, weight_path + "4096-4096.bin");
    set_weight(third_fc_filter, weight_path + "1000-4096.bin");

    set_weight(first_bias, weight_path + "64-0b.bin");
    set_weight(second_bias, weight_path + "192-1b.bin");
    set_weight(third_bias, weight_path + "384-2b.bin");
    set_weight(fourth_bias, weight_path + "256-3b.bin");
    set_weight(fifth_bias, weight_path + "256-4b.bin");

    set_weight(first_fc_bias, weight_path + "4096-5b.bin");
    set_weight(second_fc_bias, weight_path + "4096-6b.bin");
    set_weight(third_fc_bias, weight_path +"1000-7b.bin");

    int end_time = static_cast<int>(time(0));
}

void AlexNet::training_epoch(int epoch){

    for(int e = 0; e < epoch; e++){
        double avg_cost = 0;
        for(int step = 0; step < data_meta_.size(); step++){
            int start_time = static_cast<int>(time(0));
            double cost = training_batch(step);
            int end_time = static_cast<int>(time(0));

            avg_cost += cost;
            // LOGI("BatchTime : %d, step : %d", (end_time - start_time), step);
        }

        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/64-3-11-11-c.bin", first_filter, 64, 3, 11);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/192-64-5-5-c.bin", second_filter, 192, 64, 5);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/384-192-3-3-c.bin", third_filter, 384, 192, 3);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/256-384-3-3-c.bin", fourth_filter, 256, 384, 3);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/256-256-3-3-c.bin", fifth_filter, 256, 256, 3);

        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/4096-9216-c.bin", first_fc_filter, 6 * 6 * 256, 4096);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/4096-4096-c.bin", second_fc_filter, 4096, 4096);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/1000-4096-c.bin", third_fc_filter, 4096, 1000);

        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/64-0b-c.bin", first_bias,64);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/192-1b-c.bin", second_bias, 192);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/384-2b-c.bin", third_bias, 384);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/256-3b-c.bin", fourth_bias, 256);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/256-4b-c.bin", fifth_bias, 256);

        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/4096-5b-c.bin", first_fc_bias, 4096);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/4096-6b-c.bin", second_fc_bias, 4096);
        // save_weight_binary("/storage/emulated/0/FLClient/train_weight/1000-7b-c.bin", third_fc_bias, 1000);

        // LOGI("%s", "save weight!");
    }
}

double AlexNet::training_batch(int step){
//    const int office_class[] = {0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 3, 30, 31, 32, 33, 34, 35, 36, 38, 39, 4, 40, 41, 42, 43, 44, 45, 46, 47, 5, 6, 7, 8};
    string office_class[] = {"25_0,98", "25_180,100", "26_0,102", "26_180,104", "28_0,110", "28_180,112",
                                   "29_0,114", "29_180,116", "30_0,118", "30_180,120", "31_0,122", "31_180,124", "31_90,123",
                                   "32_0,126", "32_180,128", "34_0,134", "34_180,136", "35_0,138", "35_180,140", "36_0,142",
                                   "36_180,144", "36_270,145", "36_90,143", "37_180,148", "37_90,147", "38_0,150", "38_180,152",
                                   "39_0,154", "39_180,156", "39_270,157", "39_90,155", "3_0,9", "3_180,11", "3_270,12", "3_90,10",
                                   "40_0,158", "40_180,160", "40_270,161", "40_90,159"};
    double batch_cost = 0;
    for(int batch = 0; batch < batch_size_; batch++){

        int start_time = static_cast<int>(time(0));

        string path = data_meta_[batch + (step * batch_size_)].image_path;
        int cls = data_meta_[batch + (step * batch_size_)].image_class;
        string cls_str = to_string(cls);
        for(int s = 0; s < 1000; s++){
            string a = split(office_class[s], ',')[1];
            if(a == cls_str) {
                cls = s;
                break;
            }
        }

        cout << path << endl;

        Mat im = imread(path);
        // int height_center = int(im.rows / 2);
        // int width_half = int(im.cols / 2);
        // Rect rect(0, height_center - width_half, im.cols, im.cols);
        // im = im(rect);
        // resize(im, im, Size( 224, 224 ));

        double minRGB[3] = { 0.5, 0.5, 0.5 };
        double maxRGB[3] = { 0.5, 0.5, 0.5 };

        for(int r = 0; r < im.rows; r++){
            Vec3b* ptr = im.ptr<Vec3b>(r);
            for(int c = 0; c < im.cols; c++){
                input_layer[batch][0][r][c] = ((ptr[c][2] / 255.0) - minRGB[0]) / (maxRGB[0]); // r
                input_layer[batch][1][r][c] = ((ptr[c][1] / 255.0) - minRGB[1]) / (maxRGB[1]); // g
                input_layer[batch][2][r][c] = ((ptr[c][0] / 255.0) - minRGB[2]) / (maxRGB[2]); // b
            }
        }

        convolution(input_layer, first_layer, first_filter, first_bias, 4, 3, 64, 11, 224, "relu");
        vector<string> first_pooling_coord_info = max_pooling(first_layer, first_pooling, 64, 56, 3, 2);

        convolution(first_pooling, second_layer, second_filter, second_bias, 1, 64, 192, 5, 27, "relu");
        vector<string> second_pooling_coord_info = max_pooling(second_layer, second_pooling, 192, 27, 3, 2);

        convolution(second_pooling, third_layer, third_filter, third_bias, 1, 192, 384, 3, 13, "relu");

        convolution(third_layer, fourth_layer, fourth_filter, fourth_bias, 1, 384, 256, 3, 13, "relu");

        convolution(fourth_layer, fifth_layer, fifth_filter, fifth_bias, 1, 256, 256, 3, 13, "relu");
        vector<string> third_pooling_coord_info = max_pooling(fifth_layer, third_pooling, 256, 13, 3, 2);

        flatten(third_pooling, first_fc_layer, 6, 256);

        fully_connected(first_fc_layer, second_fc_layer, first_fc_filter, first_fc_bias, 4096, 6 * 6 * 256, "relu");
        fully_connected(second_fc_layer, third_fc_layer, second_fc_filter, second_fc_bias, 4096, 4096, "relu");
        fully_connected(third_fc_layer, output_layer, third_fc_filter, third_fc_bias, 1000, 4096, "none");

        /* softmax */
        double sum = 0.0;
        double min = 0.0;
        int idx = 0;
        for(int s = 0; s < 1000; s++){
            double e = exp(output_layer[batch][s]);
            sum += e;

            if(output_layer[batch][s] > min){
                min = output_layer[batch][s];
                idx = s;
            }
        }

        double soft_max[1000];
        for(int s = 0; s < 1000; s++){
            soft_max[s] =  exp(output_layer[batch][s]) / sum;
        }

        int end_time = static_cast<int>(time(0));

        double cost = -output_layer[batch][cls] + log(sum);// float cost = -log(soft_max[cla]);

        cout << cost << endl;

        batch_cost = cost;

        double error_in_1[1000];

        for(int s = 0; s < 1000; s++){
            error_in_1[s] = (1.0 / sum) * exp(output_layer[batch][s]) - (s == cls ? 1.0 : 0.0);
        }

        for(int j = 0; j < 1000; j++){
            double out_in = 1.0;
            // out_in = soft_max[j] * (1.0f - soft_max[j]);
            // out_in = 1.0f - soft_max[j];
            error_in_1[j] = error_in_1[j] * out_in;
        }

        for(int f = 0; f < 1000; f++){
            for(int j = 0; j < 4096; j++){
                double in_weihgt = third_fc_layer[batch][j];
                double gradient = error_in_1[f] * in_weihgt;
                // if(j == 0 && i < 3) cout << gradient << endl;
                // if(j == 999 && i < 3) cout << gradient << endl;
                third_fc_gradient[f][j] += gradient;
            }
            third_fc_bias_gradient[f] = error_in_1[f];
        }

        double error_in_2[4096];
        for(int f = 0; f < 4096; f++){
            double error_out_sum = 0.0;
            for(int i = 0; i < 1000; i++){
                error_out_sum += error_in_1[i] * third_fc_filter[i][f];
            }

            error_in_2[f] = error_out_sum * (third_fc_layer[batch][f] > 0 ? 1.0 : 0.0);

            for(int j = 0; j < 4096; j++){
                double in_weight = second_fc_layer[batch][j];
                double out_in = third_fc_layer[batch][f] > 0 ? 1.0 : 0.0;
                double gradient = error_out_sum * out_in * in_weight;
                // if(f == 2 && j < 3) cout << gradient << endl;
                // if(f == 4095 && j < 3) cout << gradient << endl;
                second_fc_gradient[f][j] += gradient;
            }

            second_fc_bias_gradient[f] = error_in_2[f];
        }

        double error_in_3[4096];
        for(int f = 0; f < 4096; f++){
            double error_out_sum = 0.0;
            for(int i = 0; i < 4096; i++){
                error_out_sum += error_in_2[i] * second_fc_filter[i][f];
            }

            error_in_3[f] = error_out_sum * (second_fc_layer[batch][f] > 0 ? 1.0 : 0.0);

            for(int j = 0; j < 6 * 6 * 256; j++){
                double in_weight = first_fc_layer[batch][j];
                double out_in = second_fc_layer[batch][f] > 0 ? 1.0 : 0.0;
                double gradient = error_out_sum * out_in * in_weight;
                // if(f == 1 && j < 3) cout << gradient << endl;
                // if(f == 4095 && j < 3) cout << gradient << endl;
                first_fc_gradient[f][j] += gradient;
            }

            first_fc_bias_gradient[f] = error_in_3[f];
        }


        double local_gradient[6 * 6 * 256];
        int channel_count = 0;
        int list_count = 0;
        vector<string> coord_list = split(third_pooling_coord_info[0], '\n');
        for(int f = 1; f <= 6 * 6 * 256; f++){
            double error_out_sum = 0.0;
            for(int i = 0; i < 4096; i++){
                error_out_sum += error_in_3[i] * first_fc_filter[i][f - 1];
            }

            local_gradient[f - 1] = error_out_sum * (first_fc_layer[batch][f - 1] > 0 ? 1.0 : 0.0);

            channel_count++;

            if(channel_count == 36){
                for(int m = 36; m > 0; m--){
                    vector<string> info_coord = split(coord_list[(36 - m) + 36 * list_count], '-');
                    vector<string> info = split(info_coord[0], ',');
                    vector<string> coord = split(info_coord[1], ',');

                    int pooling_channel = atoi(info[0].c_str());
                    int max_r = atoi(coord[0].c_str());
                    int max_c = atoi(coord[1].c_str());
                    delta_fifth_layer[batch][pooling_channel][max_r][max_c] += local_gradient[(36 - m) + 36 * list_count] * (fifth_layer[batch][pooling_channel][max_r][max_c] > 0 ? 1.0 : 0.0);
                }
                channel_count = 0;
                list_count++;
            }
        }

        backpropagation(fifth_filter_gradient, fifth_bias_gradient, fourth_layer, delta_fifth_layer, 256, 256, 13, 1, 3);
        get_local_gradient(delta_fourth_layer, fifth_filter, delta_fifth_layer, fourth_layer, 256, 256, 1, 3, 13);
        backpropagation(fourth_filter_gradient, fourth_bias_gradient, third_layer, delta_fourth_layer, 256, 384, 13, 1, 3);
        get_local_gradient(delta_third_layer, fourth_filter, delta_fourth_layer, third_layer, 256, 384, 1, 3, 13);
        backpropagation(third_filter_gradient, third_bias_gradient, second_pooling, delta_third_layer, 384, 192, 13, 1, 3);
        get_local_gradient(delta_second_pooling, third_filter, delta_third_layer, second_pooling, 384, 192, 1, 3, 13);
        pooling_to_layer(delta_second_pooling, delta_second_layer, second_layer, second_pooling_coord_info);
        backpropagation(second_filter_gradient, second_bias_gradient, first_pooling, delta_second_layer, 192, 64, 27, 1, 5);

        get_local_gradient(delta_first_pooling, second_filter, delta_second_layer, first_pooling, 192, 64, 1, 5, 27);
        pooling_to_layer(delta_first_pooling, delta_first_layer, first_layer, first_pooling_coord_info);
        backpropagation(first_filter_gradient, first_bias_gradient, input_layer, delta_first_layer, 64, 3, 56, 4, 11);

        weight_update(third_fc_filter, third_fc_gradient, 4096, 1000, LR);
        weight_update(second_fc_filter, second_fc_gradient, 4096, 4096, LR);
        weight_update(first_fc_filter, first_fc_gradient, 6 * 6 * 256, 4096, LR);

        weight_update(fifth_filter, fifth_filter_gradient, 256, 256, 3, LR);
        weight_update(fourth_filter, fourth_filter_gradient, 256, 384, 3, LR);
        weight_update(third_filter, third_filter_gradient, 384, 192, 3, LR);
        weight_update(second_filter, second_filter_gradient, 192, 64, 5, LR);
        weight_update(first_filter, first_filter_gradient, 64, 3, 11, LR);

        bias_update(first_bias, first_bias_gradient, 64, LR);
        bias_update(second_bias, second_bias_gradient, 192, LR);
        bias_update(third_bias, third_bias_gradient, 384, LR);
        bias_update(fourth_bias, fourth_bias_gradient, 256, LR);
        bias_update(fifth_bias, fifth_bias_gradient, 256, LR);

        bias_update(first_fc_bias, first_fc_bias_gradient, 4096, LR);
        bias_update(second_fc_bias, second_fc_bias_gradient, 4096, LR);
        bias_update(third_fc_bias, third_fc_bias_gradient, 1000, LR);

        local_gradient_zero(delta_first_layer, 64, 56);
        local_gradient_zero(delta_first_pooling, 64, 27);
        local_gradient_zero(delta_second_layer, 192, 27);
        local_gradient_zero(delta_second_pooling, 192, 13);
        local_gradient_zero(delta_third_layer, 384, 13);
        local_gradient_zero(delta_fourth_layer, 256, 13);
        local_gradient_zero(delta_fifth_layer, 256, 13);
    }

    return batch_cost;
}

void AlexNet::detroy_network(){
    /* Memory Release */
    layer_memory_release(input_layer, batch_size_, 3, 224);
    layer_memory_release(first_layer, batch_size_, 64, 56);
    layer_memory_release(first_pooling, batch_size_, 64, 27);
    layer_memory_release(second_layer, batch_size_, 192, 27);
    layer_memory_release(second_pooling, batch_size_, 192, 13);
    layer_memory_release(third_layer, batch_size_, 384, 13);
    layer_memory_release(fourth_layer, batch_size_, 256, 13);
    layer_memory_release(fifth_layer, batch_size_, 256, 13);
    layer_memory_release(third_pooling, batch_size_, 256, 6);

    layer_memory_release(first_filter, 64, 3, 11);
    layer_memory_release(second_filter, 192, 64, 5);
    layer_memory_release(third_filter, 384, 192, 3);
    layer_memory_release(fourth_filter, 256, 384, 3);
    layer_memory_release(fifth_filter, 256, 256, 3);

    delete first_bias;
    delete second_bias;
    delete third_bias;
    delete fourth_bias;
    delete fifth_bias;

    delete first_fc_bias;
    delete second_fc_bias;
    delete third_fc_bias;

    delete first_bias_gradient;
    delete second_bias_gradient;
    delete third_bias_gradient;
    delete fourth_bias_gradient;
    delete fifth_bias_gradient;

    delete first_fc_bias_gradient;
    delete second_fc_bias_gradient;
    delete third_fc_bias_gradient;

    layer_memory_release(delta_first_layer, batch_size_, 64, 56);
    layer_memory_release(delta_first_pooling, batch_size_, 64, 27);
    layer_memory_release(delta_second_layer, batch_size_, 192, 27);
    layer_memory_release(delta_second_pooling, batch_size_, 192, 13);
    layer_memory_release(delta_third_layer, batch_size_, 384, 13);
    layer_memory_release(delta_fourth_layer, batch_size_, 256, 13);
    layer_memory_release(delta_fifth_layer, batch_size_, 256, 13);

    layer_memory_release(first_filter_gradient, 64, 3, 11);
    layer_memory_release(second_filter_gradient, 192, 64, 5);
    layer_memory_release(third_filter_gradient, 384, 192, 3);
    layer_memory_release(fourth_filter_gradient, 256, 384, 3);
    layer_memory_release(fifth_filter_gradient, 256, 256, 3);

    layer_memory_release(first_fc_layer, batch_size_);
    layer_memory_release(second_fc_layer, batch_size_);
    layer_memory_release(third_fc_layer, batch_size_);
    layer_memory_release(output_layer, batch_size_);

    layer_memory_release(first_fc_filter, 4096);
    layer_memory_release(second_fc_filter, 4096);
    layer_memory_release(third_fc_filter, 1000);

    layer_memory_release(first_fc_gradient, 4096);
    layer_memory_release(second_fc_gradient, 4096);
    layer_memory_release(third_fc_gradient, 1000);

    // LOGI("Memory Release Complete!!");
}

void AlexNet::save_weight_binary(string binary_path, double ****weight, int output_channel,
                                 int input_channel, int filter_size) {
    ofstream out;
    out.open(binary_path, std::ios::out | std::ios::binary);

    for(int o = 0; o < output_channel; o++){
        for(int i = 0l; i < input_channel; i++){
            for(int r = 0; r < filter_size; r++){
                for(int c = 0; c < filter_size; c++){
                    float f = static_cast<float>(weight[o][i][r][c]);
                    out.write( reinterpret_cast<const char*>( &f ), sizeof( float ));
                }
            }
        }
    }

    out.close();
}

void AlexNet::save_weight_binary(string binary_path, double **weight, int length, int next_length) {
    ofstream out;
    out.open(binary_path, std::ios::out | std::ios::binary);

    for(int nl = 0; nl < next_length; nl++){
        for(int l = 0; l < length; l++){
            float f = static_cast<float>(weight[nl][l]);
            out.write( reinterpret_cast<const char*>( &f ), sizeof( float ));
        }
    }
    out.close();
}

void AlexNet::save_weight_binary(string binary_path, double *bias, int length) {
    ofstream out;
    out.open(binary_path, std::ios::out | std::ios::binary);

    for(int l = 0; l < length; l++){
        float f = static_cast<float>(bias[l]);
        out.write( reinterpret_cast<const char*>( &f ), sizeof( float ));
    }
    out.close();
}
