
#include <iostream>
#include <algorithm>
#include "common.h"
#include "AttrNet.h"

const string model_path = "../model/attr_metro_int8.rt";

using namespace std;

void demo_image()
{
    AttrNet* attrNet = new AttrNet();
    attrNet->loadNetFromRTModel(model_path);
    attrNet->init(96, 128, 8);

    Mat image, crop;
    for (int i = 1; i < 8; i++)
    {
        string image_path = "../data/images/image" + std::to_string(i) + ".jpg";
        image = cv::imread(image_path);
        cv::resize(image, image, cv::Size(96, 192), (0, 0), (0, 0), cv::INTER_CUBIC);
        crop = image(cv::Rect(0, 0, 96, 128));

        // inference using TensorRT model
        double start = (double) cvGetTickCount();
        Attribute result = attrNet->run(crop);
        printf("inference cost = %gms\n", ((double) cvGetTickCount() - start) / (cvGetTickFrequency() * 1000));

        cout<<"image["<<i<<"]:"<<endl;
        cout<<"Hat prob: with hat "<<result.hat[0]<<" | no hat "<<result.hat[1]<<endl;
        cout<<"Cloth prob: with cloth "<<result.cloth[0]<<" | no cloth "<<result.cloth[1]<<endl;
    }

    delete attrNet;
}

int main(int argc, char** argv)
{
    demo_image();
}
