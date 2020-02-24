CMAKE_MINIMUM_REQUIRED ( VERSION 3.8.1 )

PROJECT ( FedeNet )

SET ( MODEL "model" )

SET ( SRC
        main.cpp
        cnn.cpp
        utils.cpp
        alexnet/alexnet.cpp
        )

SET ( CMAKE_C_COMPILER "g++" )
SET ( CMAKE_CXX_FLAGS "-std=c++11 -pthread")

SET ( OpenCV_DIR /usr/local/lib/cmake/opencv4 )

ADD_EXECUTABLE( ${MODEL} ${SRC} )

FIND_PACKAGE(OpenCV REQUIRED)

TARGET_LINK_LIBRARIES (
        ${MODEL} ${OpenCV_LIBS}
        )