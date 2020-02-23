#include "cnn.hpp"

void CNN::set_data(string data_path, int batch_size, bool suffle){
    if(batch_size == 0) {
        cout << "batch size has to be over 1" << endl;
        return;
    }
    batch_size_ = batch_size;

    vector<string> img_list;
    try{
        glob(data_path + "*.*", img_list);

        for(int i = 0; i < img_list.size(); i++){
            if(img_list[i].find("png") || img_list[i].find("jpg")){
                vector<string> get_image_name = split(img_list[i], '/');
                vector<string> get_class = split(get_image_name[get_image_name.size() - 1], '-');
                InputDataInfo input_data_info;
                input_data_info.image_path = img_list[i];
                input_data_info.image_class = atoi(get_class[0].c_str());
                data_meta_.push_back(input_data_info);
            }
        }
    }catch(int exception){
        cout << exception << " : image extraction error!!" << endl;
    }

    if(data_meta_.size() != 0 && suffle){
        random_device rd;
        mt19937 g(rd());
        shuffle(data_meta_.begin(), data_meta_.end(), g);
    }
}

void CNN::convolution(double**** layer, double**** next_layer, double**** weight
                    , int stride, int input_channel, int output_channel, int filter, int size, string activ){
    for(int batch = 0; batch < batch_size_; batch++){
        int next_row = 0;
        int next_col = 0;

        int start_idx = int(filter / 2) * -1;
        int end_idx = start_idx * -1;

        for(int channel_out = 0; channel_out < output_channel; channel_out++){
            int row_idx = 0;
            int col_idx = 0;

            for(int r = 0; r < size; r = r + stride){
                for(int c = 0; c < size; c = c + stride){
                    double sum = 0;

                    for(int channel_in = 0; channel_in < input_channel; channel_in++){
                        for(int fr = start_idx; fr <= end_idx; fr++){
                            for(int fc = start_idx; fc <= end_idx; fc++){
                                if((r + fr >= 0 && c + fc >= 0) && (r + fr < size && c + fc < size)){
                                    sum += (weight[channel_out][channel_in][fr + end_idx][fc + end_idx] * layer[batch][channel_in][r + fr][c + fc]);
                                }
                            }
                        }
                    }

                    next_layer[batch][channel_out][row_idx][col_idx] = activation(sum, activ);
                    col_idx++;
                }

                row_idx++;
                
                next_row = row_idx;
                next_col = col_idx;
                col_idx = 0;
            }
        }
    }
}

vector<string> CNN::max_pooling(double**** layer, double**** pooling_layer, int output_channel, int layer_size, int filter_size, int stride){
    vector<string> ret;
    for(int batch = 0; batch < batch_size_; batch++){
        string coord_info = "";

        int next_row = 0;
        int next_col = 0;

        int start_idx = int(filter_size / 2) * -1;
        int end_idx = start_idx * -1;

        for(int channel_out = 0; channel_out < output_channel; channel_out++){
            int row_idx = 0;
            int col_idx = 0;

            for(int r = 1; r < layer_size - 1; r = r + stride){
                for(int c = 1; c < layer_size - 1; c = c + stride){
                    
                    double max = 0;
                    int max_r = r + start_idx;
                    int max_c = c + start_idx;
                    for(int fr = start_idx; fr <= end_idx; fr++){
                        for(int fc = start_idx; fc <= end_idx; fc++){
                            double value = ((r + fr >= 0 && c + fc >= 0) && (r + fr < layer_size && c + fc < layer_size) ? layer[batch][channel_out][r + fr][c + fc] : 0.0);
                            if(value > max) {
                                max = value;
                                max_r = r + fr;
                                max_c = c + fc;
                            }
                        }
                    }

                    pooling_layer[batch][channel_out][row_idx][col_idx] = max;
                    coord_info += to_string(channel_out) + "," + to_string(row_idx) + "," + to_string(col_idx) + "-" + to_string(max_r) + "," + to_string(max_c) + "\n";

                    col_idx++;
                }
                row_idx++;
                next_row = row_idx;
                next_col = col_idx;
                col_idx = 0;
            }
        }
        ret.push_back(coord_info);
    }
    return ret;
}

void CNN::flatten(double**** layer, double** fc, int layer_size, int channel){
    int fc_idx = 0;
    for(int batch = 0; batch < batch_size_; batch++){
        for(int cha = 0; cha < channel; cha++){
            for(int r = 0; r < layer_size; r++){
                for(int c = 0; c < layer_size; c++){
                    fc[batch][fc_idx] = layer[batch][cha][r][c];
                    fc_idx++;
                }
            }
        }
    }
}

double get_fc_sum(double** next_layer, double** layer, double** weight, int batch, int next_l, int length, string activ){
    double sum = 0;
    for(int l = 0; l < length; l++){
        sum += layer[batch][l] * weight[next_l][l];
    }
    next_layer[batch][next_l] = sum > 0 ? sum : 0.0;
}

void CNN::fully_connected(double** layer, double** next_layer, double** weight, int next_length, int length, string activ){
    for(int batch = 0; batch < batch_size_; batch++){
        for(int next_l = 0; next_l < next_length; next_l++){
            double sum = 0;
            for(int l = 0; l < length; l++){
                sum += layer[batch][l] * weight[next_l][l];
            }
            next_layer[batch][next_l] = activation(sum, activ);
            // thread fc_cal(&CNN::get_fc_sum, next_layer, layer, weight, next_l, length, activ);
            // fc_cal.join();
        }
    }
}

void CNN::backpropagation(double**** gradient, double**** layer, double**** delta, int output_channel, int input_channel, int size, int stride, int filter_size){
    for(int batch = 0; batch < batch_size_; batch++){
        int start_idx = int(filter_size / 2) * -1;
        int end_idx = start_idx * -1;

        for(int f = 0; f < output_channel; f++){ // 256
            for(int i = 0; i < input_channel; i++){ // 384
                // for(int r = 0; r < size; r = r + stride){
                //     for(int c = 0; c < size; c = c + stride){
                for(int r = 0; r < size; r++){
                    for(int c = 0; c < size; c++){
                        
                        for(int fr = start_idx; fr <= end_idx; fr++){
                            for(int fc = start_idx; fc <= end_idx; fc++){
                                if(((r * stride) + fr >= 0 && (c * stride) + fc >= 0) && ((r * stride) + fr < size * stride && (c * stride) + fc < size * stride)){
                                    gradient[f][i][fr + end_idx][fc + end_idx] += layer[batch][i][(r * stride) + fr][(c * stride) + fc] * delta[batch][f][r][c];
                                }
                            }
                        }
                    }
                }

                // if(f == 0 && i < 1 && filter_size == 3){
                //     cout << "[" << gradient[f][i][0][0] << " , " << gradient[f][i][0][1] << " , " << gradient[f][i][0][2] << "]" << endl;
                //     cout << "[" << gradient[f][i][1][0] << " , " << gradient[f][i][1][1] << " , " << gradient[f][i][1][2] << "]"<< endl;
                //     cout << "[" << gradient[f][i][2][0] << " , " << gradient[f][i][2][1] << " , " << gradient[f][i][2][2] << "]" << endl;
                //     cout << endl;
                // }
                // if((f == 0) && i < 1 && filter_size == 5){
                //     cout << "[" << gradient[f][i][0][0] << " , " << gradient[f][i][0][1] << " , " << gradient[f][i][0][2] << " , " << gradient[f][i][0][3] << " , " << gradient[f][i][0][4] << "]" << endl;
                //     cout << "[" << gradient[f][i][1][0] << " , " << gradient[f][i][1][1] << " , " << gradient[f][i][1][2] << " , " << gradient[f][i][1][3] << " , " << gradient[f][i][1][4] << "]" << endl;
                //     cout << "[" << gradient[f][i][2][0] << " , " << gradient[f][i][2][1] << " , " << gradient[f][i][2][2] << " , " << gradient[f][i][2][3] << " , " << gradient[f][i][2][4] << "]" << endl;
                //     cout << "[" << gradient[f][i][3][0] << " , " << gradient[f][i][3][1] << " , " << gradient[f][i][3][2] << " , " << gradient[f][i][3][3] << " , " << gradient[f][i][3][4] << "]" << endl;
                //     cout << "[" << gradient[f][i][4][0] << " , " << gradient[f][i][4][1] << " , " << gradient[f][i][4][2] << " , " << gradient[f][i][4][3] << " , " << gradient[f][i][4][4] << "]" << endl;
                //     cout << endl;
                // }

                // if(f == 0 && i < 1 && filter_size == 11){
                //     for(int z = 0; z < 11; z++){
                //         for(int x = 0; x < 11; x++){
                //             cout << gradient[f][i][z][x] << " ";
                //         }
                //         cout << endl;
                //     }
                // }
            }
        }
    }
}

void CNN::get_local_gradient(double**** local_gradient, double**** weight, double**** previous_local_gradient, double**** previous_layer, int input_channel, int output_channel, int stride, int filter_size, int size){
    for(int batch = 0; batch < batch_size_; batch++){
        int start_idx = int(filter_size / 2) * -1;
        int end_idx = start_idx * -1;
        
        for(int channel_out = 0; channel_out < output_channel; channel_out++){
            for(int channel_in = 0; channel_in < input_channel; channel_in++){
                for(int r = 0; r < size; r = r + stride){
                    for(int c = 0; c < size; c = c + stride){

                        double** weight_transpose = new double*[filter_size];
                        layer_memory_assign(weight_transpose, filter_size, filter_size);

                        transpose_filter(weight[channel_in][channel_out], weight_transpose, filter_size);

                        for(int fr = start_idx; fr <= end_idx; fr++){
                            for(int fc = start_idx; fc <= end_idx; fc++){
                                if((r + fr >= 0 && c + fc >= 0) && (r + fr < size && c + fc < size)){
                                    local_gradient[batch][channel_out][r][c] += weight_transpose[fr + end_idx][fc + end_idx] * previous_local_gradient[batch][channel_in][r + fr][c + fc];
                                    local_gradient[batch][channel_out][r][c] *= (previous_layer[batch][channel_out][r][c] > 0 ? 1.0 : 0.0);
                                }
                            }
                        }
                        layer_memory_release(weight_transpose, filter_size);
                    }
                }
            }
        }
    }
}

void CNN::pooling_to_layer(double**** delta_pooling, double**** delta_layer, double**** layer, vector<string> coord_info){
    for(int batch = 0; batch < batch_size_; batch++){
        vector<string> coord_list = split(coord_info[0], '\n');
        int length = coord_list.size();

        for(int i = 0; i < length; i++){
            vector<string> info_coord = split(coord_list[i], '-');
            vector<string> info = split(info_coord[0], ',');
            vector<string> coord = split(info_coord[1], ',');

            int pooling_channel = atoi(info[0].c_str());
            int pooling_r = atoi(info[1].c_str());
            int pooling_c = atoi(info[2].c_str());
            int max_r = atoi(coord[0].c_str());
            int max_c = atoi(coord[1].c_str());

            if(delta_pooling[batch][pooling_channel][pooling_r][pooling_c] != 0){
                delta_layer[batch][pooling_channel][max_r][max_c] += delta_pooling[batch][pooling_channel][pooling_r][pooling_c] * (layer[batch][pooling_channel][max_r][max_c] > 0 ? 1.0 : 0.0);
            }
        }
    }
}

void CNN::local_gradient_zero(double**** local_gradient, int channel_size, int size){
    for(int batch = 0; batch < batch_size_; batch++){
        for(int channel = 0; channel < channel_size; channel++){
            for(int r = 0; r < size; r++){
                for(int c = 0; c < size; c++){
                    local_gradient[batch][channel][r][c] = 0;
                }
            }
        }
    }
}

void CNN::weight_update(double**** weight, double**** gradient, int output_channel, int input_channel, int filter_size, double lr){
    for(int o = 0; o < output_channel; o++){
        for(int i = 0l; i < input_channel; i++){
            for(int r = 0; r < filter_size; r++){
                for(int c = 0; c < filter_size; c++){
                    weight[o][i][r][c] -= lr * (gradient[o][i][r][c]);

                    gradient[o][i][r][c] = 0;
                }
            }
        }
    }
}
void CNN::weight_update(double** weight, double** gradient, int length, int next_length, double lr){
    for(int nl = 0; nl < next_length; nl++){
        for(int l = 0; l < length; l++){
            weight[nl][l] -= lr * (gradient[nl][l]);

            gradient[nl][l] = 0;
        }
    }
}

double CNN::activation(double input, string activation){
    double ret = 0;
    if(activation == "relu"){
        ret = (input > 0 ? input : 0);
    }else{
        ret = input;
    }
    return ret;
}