
PROJECT(attri_inference)

INCLUDE_DIRECTORIES(${THIRD_PARTY_PATH}/tensorrt-5.1.2/include)
LINK_DIRECTORIES(/usr/local/cuda-9.0/lib64/)
LINK_DIRECTORIES(/home/xieyi/libs/cudnn_7_5/7.5.0/lib64/)
LINK_DIRECTORIES(${THIRD_PARTY_PATH}/opencv/lib)
LINK_DIRECTORIES(${THIRD_PARTY_PATH}/tensorrt-5.1.2/lib)

FIND_PACKAGE(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O0;-std=c++11;-ccbin /usr/bin/g++;
      -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52
      -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61;")


set(SOURCE_FILES
        ${common_file}
        ${infer_module_file}
        demo.cpp
        )

cuda_add_executable(${PROJECT_NAME}  ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} libcudnn.so libcublas.so libcudart_static.so
        libnvToolsExt.so libcudart.so libopencv_core.so libopencv_highgui.so libopencv_imgproc.so
        libm.so libgomp.so librt.so libpthread.so libdl.so libopencv_imgcodecs.so libopencv_videoio.so libnvinfer.so libnvcaffe_parser.so libnvinfer_plugin.so libnvonnxparser.so)
