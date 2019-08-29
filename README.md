## Resnet_to_rt

------


### requirements:

- TensorRT 5.1.0 +
- OpenCV 3.0 +
- cuda 9.0
- cudnn 7.5

### Step

1. Compile CMakeList.txt and run build_attri_rtmodel for the model convert
2. output INT8 (needs calibration images) or folat 32 model
3. run attri_inference to validate
