#include "alexnet.hpp"

void AlexNet::create_network(){
    /* AlexNet Network Memory Assign */
    input_layer = new double***[batch_size_];
    layer_memory_assign(input_layer, batch_size_, 3, 224);
    first_layer = new double***[batch_size_];
    layer_memory_assign(first_layer, batch_size_, 96, 56);
    first_pooling = new double***[batch_size_];
    layer_memory_assign(first_pooling, batch_size_, 96, 27);
    second_layer = new double***[batch_size_];
    layer_memory_assign(second_layer, batch_size_, 256, 27);
    second_pooling = new double***[batch_size_];
    layer_memory_assign(second_pooling, batch_size_, 256, 13);
    third_layer = new double***[batch_size_];
    layer_memory_assign(third_layer, batch_size_, 384, 13);
    fourth_layer = new double***[batch_size_];
    layer_memory_assign(fourth_layer, batch_size_, 384, 13);
    fifth_layer = new double***[batch_size_];
    layer_memory_assign(fifth_layer, batch_size_, 256, 13);
    third_pooling = new double***[batch_size_];
    layer_memory_assign(third_pooling, batch_size_, 256, 6);

    delta_first_layer = new double***[batch_size_];
    layer_memory_assign(delta_first_layer, batch_size_, 96, 56);
    delta_first_pooling = new double***[batch_size_];
    layer_memory_assign(delta_first_pooling, batch_size_, 96, 27);
    delta_second_layer = new double***[batch_size_];
    layer_memory_assign(delta_second_layer, batch_size_, 256, 27);
    delta_second_pooling = new double***[batch_size_];
    layer_memory_assign(delta_second_pooling, batch_size_, 256, 13);
    delta_third_layer = new double***[batch_size_];
    layer_memory_assign(delta_third_layer, batch_size_, 384, 13);
    delta_fourth_layer = new double***[batch_size_];
    layer_memory_assign(delta_fourth_layer, batch_size_, 384, 13);
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

    first_filter = new double***[96];
    layer_memory_assign(first_filter, 96, 3, 11);
    second_filter = new double***[256];
    layer_memory_assign(second_filter, 256, 96, 5);
    third_filter = new double***[384];
    layer_memory_assign(third_filter, 384, 256, 3);
    fourth_filter = new double***[384];
    layer_memory_assign(fourth_filter, 384, 384, 3);
    fifth_filter = new double***[256];
    layer_memory_assign(fifth_filter, 256, 384, 3);

    first_fc_filter = new double*[4096];
    layer_memory_assign(first_fc_filter, 4096, 6 * 6 * 256);
    second_fc_filter = new double*[4096];
    layer_memory_assign(second_fc_filter, 4096, 4096);
    third_fc_filter = new double*[1000];
    layer_memory_assign(third_fc_filter, 1000, 4096);

    first_filter_gradient = new double***[96];
    layer_memory_assign(first_filter_gradient, 96, 3, 11);
    second_filter_gradient = new double***[256];
    layer_memory_assign(second_filter_gradient, 256, 96, 5);
    third_filter_gradient = new double***[384];
    layer_memory_assign(third_filter_gradient, 384, 256, 3);
    fourth_filter_gradient = new double***[384];
    layer_memory_assign(fourth_filter_gradient, 384, 384, 3);
    fifth_filter_gradient = new double***[256];
    layer_memory_assign(fifth_filter_gradient, 256, 384, 3);

    first_fc_gradient = new double*[4096];
    layer_memory_assign(first_fc_gradient, 4096, 6 * 6 * 256);
    second_fc_gradient = new double*[4096];
    layer_memory_assign(second_fc_gradient, 4096, 4096);
    third_fc_gradient = new double*[1000];
    layer_memory_assign(third_fc_gradient, 1000, 4096);
    
    cout << "Memory Assign Complete!!" << endl;
}

void AlexNet::set_trainig(double learning_rate, string optimizer, string weight_path){
    LR = learning_rate;    
    set_weight(first_filter, weight_path + "96-3-11-11.txt");
    set_weight(second_filter, weight_path + "256-96-5-5.txt");
    set_weight(third_filter, weight_path + "384-256-3-3.txt");
    set_weight(fourth_filter, weight_path + "384-384-3-3.txt");
    set_weight(fifth_filter, weight_path + "256-384-3-3.txt");

    set_weight(first_fc_filter, weight_path + "4096-9216.txt");
    set_weight(second_fc_filter, weight_path + "4096-4096.txt");
    set_weight(third_fc_filter, weight_path + "1000-4096.txt");
}

void AlexNet::training_epoch(int epoch){
    for(int e = 0; e < epoch; e++){
        double avg_cost = 0;
        int start_time = time(0);
        for(int step = 0; step < data_meta_.size(); step++){
            double cost = training_batch(step);
            avg_cost += cost;
            cout << "COST : " << cost << endl;
        }
        int end_time = time(0);
        cout << " EPOCH : " << (e + 1) << " (" << (end_time - start_time) << "second) " << "LOSS : " << (avg_cost / data_meta_.size()) << endl;
    }
}

double AlexNet::training_batch(int step){
    double batch_cost = 0;
    for(int batch = 0; batch < batch_size_; batch++){

        string path = data_meta_[batch + (step * batch_size_)].image_path;
        int cls = data_meta_[batch + (step * batch_size_)].image_class;
        Mat im = imread(path);

        for(int r = 0; r < im.rows; r++){
            Vec3b* ptr = im.ptr<Vec3b>(r);
            for(int c = 0; c < im.cols; c++){
                input_layer[batch][0][r][c] = ptr[c][2]; // r
                input_layer[batch][1][r][c] = ptr[c][1]; // g
                input_layer[batch][2][r][c] = ptr[c][0]; // b
            }
        }

        convolution(input_layer, first_layer, first_filter, 4, 3, 96, 11, 224, "relu");
        vector<string> first_pooling_coord_info = max_pooling(first_layer, first_pooling, 96, 56, 3, 2);

        convolution(first_pooling, second_layer, second_filter, 1, 96, 256, 5, 27, "relu");
        vector<string> second_pooling_coord_info = max_pooling(second_layer, second_pooling, 256, 27, 3, 2);

        convolution(second_pooling, third_layer, third_filter, 1, 256, 384, 3, 13, "relu");
        convolution(third_layer, fourth_layer, fourth_filter, 1, 384, 384, 3, 13, "relu");
        convolution(fourth_layer, fifth_layer, fifth_filter, 1, 384, 256, 3, 13, "relu");
        vector<string> third_pooling_coord_info = max_pooling(fifth_layer, third_pooling, 256, 13, 3, 2);

        flatten(third_pooling, first_fc_layer, 6, 256);

        fully_connected(first_fc_layer, second_fc_layer, first_fc_filter, 4096, 6 * 6 * 256, "relu");
        fully_connected(second_fc_layer, third_fc_layer, second_fc_filter, 4096, 4096, "relu");
        fully_connected(third_fc_layer, output_layer, third_fc_filter, 1000, 4096, "none");

        /* softmax */
        double sum = 0.0;
        for(int s = 0; s < 1000; s++){
            double e = exp(output_layer[batch][s]);
            sum += e;
        }

        double cost = -output_layer[batch][cls] + log(sum);// float cost = -log(soft_max[cla]);
        batch_cost = cost;

        // cout << "LOSS : " << cost << endl;

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

        backpropagation(fifth_filter_gradient, fourth_layer, delta_fifth_layer, 256, 384, 13, 1, 3);
        get_local_gradient(delta_fourth_layer, fifth_filter, delta_fifth_layer, fourth_layer, 256, 384, 1, 3, 13);
        backpropagation(fourth_filter_gradient, third_layer, delta_fourth_layer, 384, 384, 13, 1, 3);
        get_local_gradient(delta_third_layer, fourth_filter, delta_fourth_layer, third_layer, 384, 384, 1, 3, 13);
        backpropagation(third_filter_gradient, second_pooling, delta_third_layer, 384, 256, 13, 1, 3);
        get_local_gradient(delta_second_pooling, third_filter, delta_third_layer, second_pooling, 384, 256, 1, 3, 13);
        pooling_to_layer(delta_second_pooling, delta_second_layer, second_layer, second_pooling_coord_info);
        backpropagation(second_filter_gradient, first_pooling, delta_second_layer, 256, 96, 27, 1, 5);

        get_local_gradient(delta_first_pooling, second_filter, delta_second_layer, first_pooling, 256, 96, 1, 5, 27);
        pooling_to_layer(delta_first_pooling, delta_first_layer, first_layer, first_pooling_coord_info);
        backpropagation(first_filter_gradient, input_layer, delta_first_layer, 96, 3, 56, 4, 11);

        weight_update(third_fc_filter, third_fc_gradient, 4096, 1000, LR);
        weight_update(second_fc_filter, second_fc_gradient, 4096, 4096, LR);
        weight_update(first_fc_filter, first_fc_gradient, 6 * 6 * 256, 4096, LR);

        weight_update(fifth_filter, fifth_filter_gradient, 256, 384, 3, LR);
        weight_update(fourth_filter, fourth_filter_gradient, 384, 384, 3, LR);
        weight_update(third_filter, third_filter_gradient, 384, 256, 3, LR);
        weight_update(second_filter, second_filter_gradient, 256, 96, 5, LR);
        weight_update(first_filter, first_filter_gradient, 96, 3, 11, LR);

        local_gradient_zero(delta_first_layer, 96, 56);
        local_gradient_zero(delta_first_pooling, 96, 27);
        local_gradient_zero(delta_second_layer, 256, 27);
        local_gradient_zero(delta_second_pooling, 256, 13);
        local_gradient_zero(delta_third_layer, 384, 13);
        local_gradient_zero(delta_fourth_layer, 384, 13);
        local_gradient_zero(delta_fifth_layer, 256, 13);
    }

    return batch_cost;
}

void AlexNet::detroy_network(){
    /* Memory Release */
    layer_memory_release(input_layer, batch_size_, 3, 224);
    layer_memory_release(first_layer, batch_size_, 96, 56);
    layer_memory_release(first_pooling, batch_size_, 96, 27);
    layer_memory_release(second_layer, batch_size_, 256, 27);
    layer_memory_release(second_pooling, batch_size_, 256, 13);
    layer_memory_release(third_layer, batch_size_, 384, 13);
    layer_memory_release(fourth_layer, batch_size_, 384, 13);
    layer_memory_release(fifth_layer, batch_size_, 256, 13);
    layer_memory_release(third_pooling, batch_size_, 256, 6);

    layer_memory_release(first_filter, 96, 3, 11);
    layer_memory_release(second_filter, 256, 96, 5);
    layer_memory_release(third_filter, 384, 256, 3);
    layer_memory_release(fourth_filter, 384, 384, 3);
    layer_memory_release(fifth_filter, 256, 384, 3);

    layer_memory_release(delta_first_layer, batch_size_, 96, 56);
    layer_memory_release(delta_first_pooling, batch_size_, 96, 27);
    layer_memory_release(delta_second_layer, batch_size_, 256, 27);
    layer_memory_release(delta_second_pooling, batch_size_, 256, 13);
    layer_memory_release(delta_third_layer, batch_size_, 384, 13);
    layer_memory_release(delta_fourth_layer, batch_size_, 384, 13);
    layer_memory_release(delta_fifth_layer, batch_size_, 256, 13);

    layer_memory_release(first_filter_gradient, 96, 3, 11);
    layer_memory_release(second_filter_gradient, 256, 96, 5);
    layer_memory_release(third_filter_gradient, 384, 256, 3);
    layer_memory_release(fourth_filter_gradient, 384, 384, 3);
    layer_memory_release(fifth_filter_gradient, 256, 384, 3);

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

    cout << "Memory Release Complete!!" << endl;
}