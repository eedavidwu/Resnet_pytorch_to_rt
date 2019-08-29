## Resnet_to_rt

------


### requirements:

- TensorRT 5.1.0 +
- OpenCV 3.0 +
- cuda 9.0
- cudnn 7.5

### Step

1. 编译CMakeList.txt，然后运行build_attri_rtmodel进行TensorRT模型转换
2. 可以选择输出INT8模型或FP32模型，INT8模型需要提供calibration images
3. 运行attri_inference进行模型推理结果的验证
