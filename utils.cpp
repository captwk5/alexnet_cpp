#include "cnn.hpp"

vector<string> CNN::split(string str, char delimiter){
    vector<string> internal;
	stringstream ss(str);
	string temp;

	while (getline(ss, temp, delimiter)) {
		internal.push_back(temp);
	}

	return internal;
}


void CNN::set_weight(double**** weight, string weight_path){
    std::ifstream i(weight_path);
    json j;
    i >> j;

    uint64_t s = j.size();

    for(int i = 0; i < s; i++){
        uint64_t s1 = j[i].size();
        for(int q = 0; q < s1; q++){
            uint64_t s2 = j[i][q].size();
            for(int k = 0; k < s2; k++){
                uint64_t s3 = j[i][q][k].size();
                for(int l = 0; l < s3; l++){
                    weight[i][q][k][l] = j[i][q][k][l];
                }
            }
        }
    }

    /**
    string str_buf_layer;
    fstream fs_layer;
    try{
        fs_layer.open(weight_path.c_str(), ios::in);
        while(!fs_layer.eof()){
            getline(fs_layer, str_buf_layer, '\n');
            vector<string> w = split(str_buf_layer, ',');
            if(w.size() == 2){
                vector<string> m = split(w[0], '-');
                int i0 = atoi(m[0].c_str());
                int i1 = atoi(m[1].c_str());
                int i2 = atoi(m[2].c_str());
                int i3 = atoi(m[3].c_str());
                double value = atof(w[1].c_str());
                weight[i0][i1][i2][i3] = value;
            }
        }
        fs_layer.close();
    }catch(int exception){

    }
    **/
}

void CNN::set_weight(double** weight, string weight_path){
    std::ifstream i(weight_path);
    json j;
    i >> j;

    uint64_t s = j.size();

    for(int i = 0; i < s; i++){
        uint64_t s1 = j[i].size();
        for(int q = 0; q < s1; q++){
            weight[i][q] = j[i][q];
        }
    }

    /**
    string str_buf_layer;
    fstream fs_layer;
    try{
        fs_layer.open(weight_path.c_str(), ios::in);
        while(!fs_layer.eof()){
            getline(fs_layer, str_buf_layer, '\n');
            vector<string> w = split(str_buf_layer, ',');
            if(w.size() == 2){
                vector<string> m = split(w[0], '-');
                int i0 = atoi(m[0].c_str());
                int i1 = atoi(m[1].c_str());
                double value = atof(w[1].c_str());
                weight[i0][i1] = value;
            }
        }
        fs_layer.close();
    }catch(int exception){

    }
    **/
}

void CNN::transpose_filter(double** weight, double** transpose_weight, int kernel_size){
    int kernel_size_idx = kernel_size - 1;
    for(int r = 0; r < kernel_size; r++){
        for(int c = 0; c < kernel_size; c++){
            transpose_weight[r][c] = weight[kernel_size_idx - r][kernel_size_idx - c];
        }
    }
}

void CNN::layer_memory_assign(double**** layer, int batch_size, int channels, int size){
    for(int i = 0; i < batch_size; i++){
        *(layer + i) = new double**[channels];
        for(int j = 0; j < channels; j++){
            *(*(layer + i) + j) = new double*[size];
            for(int k = 0; k < size; k++){
                *(*(*(layer + i) + j) + k) = new double[size];
            }
        }
    }
}

void CNN::layer_memory_assign(char**** layer, int batch_size, int channels, int size){
    for(int i = 0; i < batch_size; i++){
        *(layer + i) = new char**[channels];
        for(int j = 0; j < channels; j++){
            *(*(layer + i) + j) = new char*[size];
            for(int k = 0; k < size; k++){
                *(*(*(layer + i) + j) + k) = new char[size];
            }
        }
    }
}

void CNN::layer_memory_assign(double*** layer, int batch_size, int length, int next_length){
    for(int i = 0; i < batch_size; i++){
        *(layer + i) = new double*[length];
        for(int j = 0; j < length; j++){
            *(*(layer + i) + j) = new double[next_length];
        }
    }
}

void CNN::layer_memory_release(double**** layer, int batch_size, int channel, int size){
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < channel; j++){
            for(int k = 0; k < size; k++){
                delete[] *(*(*(layer + i) + j) + k);
            }
        }
    }
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < channel; j++){
            delete[] *(*(layer + i) + j);
        }
    }
    for(int i = 0; i < batch_size; i++){
        delete[] *(layer + i);
    }
    delete[] layer;
}

void CNN::layer_memory_release(char**** layer, int batch_size, int channel, int size){
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < channel; j++){
            for(int k = 0; k < size; k++){
                delete[] *(*(*(layer + i) + j) + k);
            }
        }
    }
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < channel; j++){
            delete[] *(*(layer + i) + j);
        }
    }
    for(int i = 0; i < batch_size; i++){
        delete[] *(layer + i);
    }
    delete[] layer;
}

void CNN::layer_memory_release(double*** layer, int batch_size, int length){
    for(int i = 0; i < batch_size; i++){
        for(int j = 0; j < length; j++){
            delete[] *(*(layer + i) + j);
        }
    } 
    for(int i = 0; i < batch_size; i++){
        delete[] *(layer + i);
    }
    delete[] layer;
}

void CNN::layer_memory_assign(double** layer, int batch_size, int length){
    for(int batch = 0; batch < batch_size; batch++){
        *(layer + batch) = new double[length];
    }
}

void CNN::layer_memory_release(double** layer, int batch_size){
    for(int i = 0; i < batch_size; i++){
        delete[] *(layer + i);
    }
    delete[] layer;
}

// vector<string> split_test(string str, char delimiter){
//     vector<string> internal;
// 	stringstream ss(str);
// 	string temp;

// 	while (getline(ss, temp, delimiter)) {
// 		internal.push_back(temp);
// 	}

// 	return internal;
// }

// void set_weight_test(double**** weight, string weight_path){
//     string str_buf_layer;
//     fstream fs_layer;
//     try{
//         fs_layer.open(weight_path.c_str(), ios::in);
//         while(!fs_layer.eof()){
//             getline(fs_layer, str_buf_layer, '\n');
//             vector<string> w = split_test(str_buf_layer, ',');
//             if(w.size() == 2){
//                 vector<string> m = split_test(w[0], '-');
//                 int i0 = atoi(m[0].c_str());
//                 int i1 = atoi(m[1].c_str());
//                 int i2 = atoi(m[2].c_str());
//                 int i3 = atoi(m[3].c_str());
//                 double value = atof(w[1].c_str());
//                 weight[i0][i1][i2][i3] = value;
//             }
//         }
//         fs_layer.close();
//     }catch(int exception){

//     }
// }

// void set_weight_test2(double** weight, string weight_path){
//     string str_buf_layer;
//     fstream fs_layer;
//     try{
//         fs_layer.open(weight_path.c_str(), ios::in);
//         while(!fs_layer.eof()){
//             getline(fs_layer, str_buf_layer, '\n');
//             vector<string> w = split_test(str_buf_layer, ',');
//             if(w.size() == 2){
//                 vector<string> m = split_test(w[0], '-');
//                 int i0 = atoi(m[0].c_str());
//                 int i1 = atoi(m[1].c_str());
//                 double value = atof(w[1].c_str());
//                 weight[i0][i1] = value;
//             }
//         }
//         fs_layer.close();
//     }catch(int exception){

//     }
// }
