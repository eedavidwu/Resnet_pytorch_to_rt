#ifndef RT_UTILS_H
#define RT_UTILS_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include <string>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <dirent.h>
#include <ftw.h>
#include <fnmatch.h>

using namespace cv;
using namespace std;

void image2MatrixBatch(const Mat &image, float *input, int startIndex, const vector<float> &meanValue = {0, 0, 0}, const float normFactor = 1);
void image2MatrixBatchF32(const Mat &image, float *input, int startIndex, const vector<float> &meanValue = {0, 0, 0}, const float normFactor = 1);
void image2MatrixBatchYolo(const Mat &image, const int net_width,  const int net_height,  float *input, int startIndex);
void mat2pic(cv::Mat &cvmat, float *data, int start, const vector<float> &MeanValue, const vector<float> &StdValue);
void getFilesRecursive(const string &file_dir, vector<string> &filepaths);
void randProduce(string filedir, vector<string> &filepaths, int num);

class DirImageScan {
    static vector<string> impaths;
    static vector<string> filters;

    //use class member as callback, set it as static
    static int callback(const char *fpath, const struct stat *sb, int typeflag);

public:
    DirImageScan();

    bool scan(string image_dir);

    vector<string> get_impaths();

    ~DirImageScan();
};


#endif //UTILS_H
