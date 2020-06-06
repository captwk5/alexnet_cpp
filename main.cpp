#include <iostream>
#include <thread>
#include <vector>
#include "alexnet/alexnet.hpp"

using namespace std;

int main(int argv, char** argc){
    AlexNet* alexnet = new AlexNet(); 

    delete alexnet;

    return 0;
}
