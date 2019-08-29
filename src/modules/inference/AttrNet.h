//
// Created by xieyi on 11/9/18.
//

#ifndef ATTR_NET_H
#define ATTR_NET_H

#include "BaseNet.h"

class AttrNet : public BaseNet
{
public:
    AttrNet();
    virtual void init(int width, int height, int maxBacthSize);
    virtual ~AttrNet();
    Attribute run(cv::Mat &img);
    void convertData(cv::Mat &img, int i);

private:
    void clean();
    void image2RGBMatrixBatch(const cv::Mat &image, float *input, int startIndex);

private:
    vector<float> mean_value{0.485, 0.456, 0.406};
    vector<float> std_value{0.229, 0.224, 0.225};
};

#endif //ATTR_NET_H
