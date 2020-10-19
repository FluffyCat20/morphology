//#include <QCoreApplication>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

void VanHerkInit(const cv::Mat& im, bool flag, const int a, const int b, std::vector<std::vector<uchar>>& rd, std::vector<std::vector<uchar>>& ru, std::vector<std::vector<uchar>>& ld, std::vector<std::vector<uchar>>& lu)
//flag: true = max, false = min
//rd, ru, ld, lu <=> rightDown, rightUp, leftDown, leftUp - max/min search direction
{
    if (im.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");
    int Ny = im.rows / (a - 1), Nx = im.cols / (b - 1);

    for (int i = 0; i < (a-1)*Ny; i+=a-1)
        for (int j = 0; j < (b-1)*Nx; j+=b-1)
        {
            uchar* pixel = im.data + (i + a - 2) * im.step + j + b - 2;//right down corner of the rectangle
            {rd[i + a - 2][j + b - 2] = pixel[0];
            for (int i1 = i + a - 3; i1 > i - 1; i1--)
            {
                pixel -= im.step;
                if (flag) rd[i1][j + b - 2] = std::max(pixel[0], (pixel + im.step)[0]);
                    else rd[i1][j + b - 2] = std::min(pixel[0], (pixel + im.step)[0]);
            } //right colomn
            pixel = im.data + (i + a - 2) * im.step + j + b - 2;
            for (int j1 = j + b - 3; j1 > j - 1; j1--)
            {
                pixel -= 1;
                if (flag) rd[i + a - 2][j1] = std::max(pixel[0], pixel[1]);
                    else rd[i + a - 2][j1] = std::min(pixel[0], pixel[1]);
            }//bottom row
            for (int i1 = i + a - 3; i1 > i - 1; i1--)
                for (int j1 = j + b - 3; j1 > j - 1; j1--)
                {
                    pixel = im.data + i1 * im.step + j1;
                    if (flag) rd[i1][j1] = std::max(std::max(pixel[0], pixel[1]), (pixel + im.step)[0]);
                        else rd[i1][j1] = std::min(std::min(pixel[0], pixel[1]), (pixel + im.step)[0]);
                }
            }

            {pixel = im.data + i * im.step + j + b - 2;//right up corner
            ru[i][j + b - 2] = pixel[0];
            for (int i1 = i + 1; i1 < i + a - 1; i1++)
            {
                pixel += im.step;
                if (flag) ru[i1][j + b - 2] = std::max(pixel[0], (pixel - im.step)[0]);
                    else ru[i1][j + b - 2] = std::min(pixel[0], (pixel - im.step)[0]);
            }//right colomn
            pixel = im.data + i * im.step + j + b - 2;
            for (int j1 = j + b - 3; j1 > j - 1; j1--)
            {
                pixel -= 1;
                if (flag) ru[i][j1] = std::max(pixel[0], pixel[1]);
                    else ru[i][j1] = std::min(pixel[0], pixel[1]);
            }//up row
            for (int i1 = i + 1; i1 < i + a - 1; i1++)
                for (int j1 = j + b - 3; j1 > j - 1; j1--)
                {
                    pixel = im.data + i1 * im.step + j1;
                    if (flag) ru[i1][j1] = std::max(std::max(pixel[0], pixel[1]), (pixel - im.step)[0]);
                        else ru[i1][j1] = std::min(std::min(pixel[0], pixel[1]), (pixel - im.step)[0]);
                }
            }

            {pixel = im.data + (i + a - 2) * im.step + j; //left down corner
            ld[i + a - 2][j] = pixel[0];
            for (int i1 = i + a - 3; i1 > i - 1; i1--)
            {
                pixel -= im.step;
                if (flag) ld[i1][j] = std::max(pixel[0], (pixel + im.step)[0]);
                    else ld[i1][j] = std::min(pixel[0], (pixel + im.step)[0]);
            }//left colomn
            pixel = im.data + (i + a - 2) * im.step + j;
            for (int j1 = j + 1; j1 < j + b - 1; j1++)
            {
                pixel += 1;
                if (flag) ld[i + a - 2][j1] = std::max(pixel[0], (pixel - 1)[0]);
                    else ld[i + a - 2][j1] = std::min(pixel[0], (pixel - 1)[0]);
            }//down row
            for (int i1 = i + a - 3; i1 > i - 1; i1--)
                for (int j1 = j + 1; j1 < j + b - 1; j1++)
                {
                    pixel = im.data + i1 * im.step + j1;
                    if (flag) ld[i1][j1] = std::max(std::max(pixel[0], (pixel - 1)[0]), (pixel + im.step)[0]);
                        else ld[i1][j1] = std::min(std::min(pixel[0], (pixel - 1)[0]), (pixel + im.step)[0]);
                }
            }

            {pixel = im.data + i * im.step + j; //left up corner
            lu[i][j] = pixel[0];
            for (int i1 = i + 1; i1 < i + a - 1; i1++)
            {
                pixel += im.step;
                if (flag) lu[i1][j] = std::max(pixel[0], (pixel - im.step)[0]);
                    else lu[i1][j] = std::min(pixel[0], (pixel - im.step)[0]);
            }//left colomn
            pixel = im.data + i * im.step + j;
            for (int j1 = j + 1; j1 < j + b - 1; j1++)
            {
                pixel += 1;
                if (flag) lu[i][j1] = std::max(pixel[0], (pixel - 1)[0]);
                    else lu[i][j1] = std::min(pixel[0], (pixel - 1)[0]);
            }//up row
            for (int i1 = i + 1; i1 < i + a - 1; i1++)
                for (int j1 = j + 1; j1 < j + b - 1; j1++)
                {
                    pixel = im.data + i1 * im.step + j1;
                    if (flag) lu[i1][j1] = std::max(std::max(pixel[0], (pixel - 1)[0]), (pixel - im.step)[0]);
                        else lu[i1][j1] = std::min(std::min(pixel[0], (pixel - 1)[0]), (pixel - im.step)[0]);
                }
            }
        }


}

uchar VanHerk(bool flag, int i, int j, const int a, const int b, const int sizeI, const int sizeJ, const std::vector<std::vector<uchar>>& rd, const std::vector<std::vector<uchar>>& ru, const std::vector<std::vector<uchar>>& ld, const std::vector<std::vector<uchar>>& lu)
//flag: true = max, false = min
//i, j - a pixel to draw
//a*b - rectangle (morphology element) size
//sizeI*sizeJ - image size
{
    int up = std::max(0, i - a / 2), down = std::min(sizeI-1, i + a / 2), left = std::max(0, j - b / 2), right = std::min(sizeJ-1, j + b / 2);
    if (flag)
    {
        return std::max(std::max (ld[up][right], lu[down][right]), std::max(rd[up][left], ru[down][left]));
    }
    else
    {
        return std::min(std::min(ld[up][right], lu[down][right]), std::min(rd[up][left], ru[down][left]));
    }
}

void delatation(cv::Mat& image, int rectSizeI, int rectSizeJ, const std::vector<std::vector<uchar>>& v1, const std::vector<std::vector<uchar>>& v2, const std::vector<std::vector<uchar>>& v3, const std::vector<std::vector<uchar>>& v4)
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");
    for (int i = 0; i < image.rows; i++)
    {
        uchar* line = image.data + i * image.step;
        for (int j = 0; j < image.cols; j++)
            line[j] = VanHerk(1, i, j, rectSizeI, rectSizeJ, image.rows, image.cols, v1, v2, v3, v4);
    }
}

void erosion(cv::Mat &image, int rectSizeI, int rectSizeJ, const std::vector<std::vector<uchar>>& v1, const std::vector<std::vector<uchar>>& v2, const std::vector<std::vector<uchar>>& v3, const std::vector<std::vector<uchar>>& v4)
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");
    for (int i = 0; i < image.rows; i++)
    {
        uchar* line = image.data + i * image.step;
        for (int j = 0; j < image.cols; j++)
            line[j] = VanHerk(0, i, j, rectSizeI, rectSizeJ, image.rows, image.cols, v1, v2, v3, v4);
    }
}

int main(int argc, char *argv[])
{
    //QCoreApplication a(argc, argv);
    int d=15; //must be odd!
    std::cout << "Hello world" << std::endl;
    cv::Mat img = cv::imread( argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cout << "error" << std::endl;
        return -1;
    }
    if (img.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    //initialisation
    std::vector<std::vector<uchar>>
        maxRD(img.rows, std::vector <uchar>(img.cols, 0)),
        maxRU(img.rows, std::vector <uchar>(img.cols, 0)),
        maxLU(img.rows, std::vector <uchar>(img.cols, 0)),
        maxLD(img.rows, std::vector <uchar>(img.cols, 0)),
        minRD(img.rows, std::vector <uchar>(img.cols, 0)),
        minRU(img.rows, std::vector <uchar>(img.cols, 0)),
        minLU(img.rows, std::vector <uchar>(img.cols, 0)),
        minLD(img.rows, std::vector <uchar>(img.cols, 0));

    VanHerkInit(img, 1, d, d, maxRD, maxRU, maxLD, maxLU);
    VanHerkInit(img, 0, d, d, minRD, minRU, minLD, minLU);

    /*for (int i=0; i<img.rows; i++)
    {
        for (int j=0; j<img.cols; j++)
            std::cout << int(maxLU[i][j]) << ' ';
        std::cout << std::endl;
    }*/

    cv::Mat result (img);
    //delatation(result, d, d, maxRD, maxRU, maxLD, maxLU);
    erosion (result, d, d, minRD, minRU, minLD, minLU);

    /*for (int i = 0; i < result.rows; i++)
    {
        uchar* line = result.data + i * result.step;
        for (int j = 0; j < result.cols; j++)
            line[j] = maxRU[i][j];
    }*/

    cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
    cv::imshow("window", result);
    cv::waitKey(0);

    cv::destroyWindow("window");
    return 0;
}
