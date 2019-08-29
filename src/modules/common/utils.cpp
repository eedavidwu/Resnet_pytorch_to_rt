#include "utils.h"

void image2MatrixBatch(const Mat &image, float *input, int startIndex, const vector<float> &meanValue,
                       const float normFactor)
{
    if ((image.data == NULL) || (image.type() != CV_8UC3)) {
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }
    if (input == NULL) {
        return;
    }

    float *p = input + startIndex;
    for (int rowI = 0; rowI < image.rows; rowI++) {
        for (int colK = 0; colK < image.cols; colK++) {
            *p = (float(image.at<Vec3b>(rowI, colK)[2]) - meanValue[0]) / normFactor;
            *(p + image.rows * image.cols) = (float(image.at<Vec3b>(rowI, colK)[1]) - meanValue[1]) / normFactor;
            *(p + 2 * image.rows * image.cols) =
                    (float(image.at<Vec3b>(rowI, colK)[0]) - meanValue[2]) / normFactor;
            p++;
        }
    }
}

void image2MatrixBatchF32(const Mat &image, float *input, int startIndex, const vector<float> &meanValue, const float normFactor)
{
    if ((image.data == NULL) || (image.type() != CV_32FC3)) {
        cout << "image's type is wrong!!Please set CV_32FC3" << endl;
        return;
    }
    if (input == NULL) {
        return;
    }

    int index = startIndex;
    for (int rowI = 0; rowI < image.rows; rowI++) {
        for (int colK = 0; colK < image.cols; colK++) {
            input[index] = image.at<Vec3f>(rowI, colK)[0];
            input[index + image.rows * image.cols] = image.at<Vec3f>(rowI, colK)[1];
            input[index + 2 * image.rows * image.cols] = image.at<Vec3f>(rowI, colK)[2];
            index++;
        }
    }
}

void image2MatrixBatchYolo(const Mat &image, const int net_width, const int net_height, float *input, int startIndex) {
    if ((image.data == NULL) || (image.type() != CV_8UC3)) {
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }

    int im_width = image.cols;
    int im_height = image.rows;
    int shift_w = (net_width - im_width) / 2;
    int shift_h = (net_height - im_height) / 2;

    if (net_width < im_width || net_height < im_height) {
        cout << "image size larger than Net size" << endl;
        return;
    }

    if (input == NULL) {
        return;
    }

    float *p = input + startIndex;
    for (int rowI = 0; rowI < net_height; rowI++) {
        for (int colK = 0; colK < net_width; colK++) {
            if (rowI >= shift_h && rowI < im_height + shift_h && colK >= shift_w && colK < im_width + shift_w) {
                *p = (float) (image.at<cv::Vec3b>(rowI - shift_h, colK - shift_w)[0]) / 255.0f;
                *(p + net_width * net_height) =
                        (float) (image.at<cv::Vec3b>(rowI - shift_h, colK - shift_w)[1]) / 255.0f;
                *(p + 2 * net_width * net_height) =
                        (float) (image.at<cv::Vec3b>(rowI - shift_h, colK - shift_w)[2]) / 255.0f;
            } else {
                *p = 0.4980392156f;
                *(p + net_width * net_height) = 0.0f;
                *(p + 2 * net_width * net_height) = 0.0f;
            }
            p++;
        }
    }
}


void mat2pic(cv::Mat &cvmat, float *data, int start, const vector<float> &MeanValue, const vector<float> &StdValue)
{
    IplImage iplimg = IplImage(cvmat);
    unsigned char *ImgData = (unsigned char *) iplimg.imageData;
    int h = iplimg.height;
    int w = iplimg.width;
    int c = iplimg.nChannels;
    int step = iplimg.widthStep;
    int i, j, k;

    for (k = 0; k < c; ++k) {
        int index = k * w * h;
        for (i = 0; i < h; ++i) {
            for (j = 0; j < w; ++j) {
                data[start + index + i * w + j] = (float(ImgData[i * step + j * c + k]) / 255.0 - MeanValue[k]) / StdValue[k];
            }
        }
    }
}


void getFilesRecursive(const string &file_dir, vector<string> &filepaths) {
    DirImageScan dirImageScan = DirImageScan();
    dirImageScan.scan(file_dir);
    vector<string> filepaths_scan = dirImageScan.get_impaths();
    for (int i = 0; i < filepaths_scan.size(); ++i) {
        filepaths.push_back(filepaths_scan[i]);
    }
    filepaths_scan.clear();
}


void randProduce(string filedir, vector<string> &filepaths, int num) {
    filepaths.clear();
    srand(unsigned(time(NULL)));
    getFilesRecursive(filedir, filepaths);
    std::random_shuffle(filepaths.begin(), filepaths.end());
    while (filepaths.size() > num) {
        filepaths.pop_back();
    }
}

//DirImageScan implementation
vector<string> DirImageScan::impaths{};
vector<string> DirImageScan::filters{"*.jpg"};

int DirImageScan::callback(const char *fpath, const struct stat *sb, int typeflag) {
    if (typeflag == FTW_F) {
        int i;
        //for each filter
        for (i = 0; i < sizeof(filters) / sizeof(filters[0]); ++i) {
            //if the filename matches the filter
            if (fnmatch(filters[i].c_str(), fpath, FNM_CASEFOLD) == 0) {
                impaths.push_back(fpath);
                //printf("found image: %s\n", fpath);
                break;
            }
        }
    }
    return 0;
}

DirImageScan::DirImageScan() {
}

bool DirImageScan::scan(string image_dir) {
    impaths.clear();
    ftw(image_dir.c_str(), callback, 16);
}

vector<string> DirImageScan::get_impaths() {
    return impaths;
}

DirImageScan::~DirImageScan() {};
