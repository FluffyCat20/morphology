#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

void VanHerkInit(const cv::Mat& im, bool flag, const int StringNumber, const int a, std::vector<uchar> &l, std::vector<uchar> &r)
//if max bool = 1, if min bool = 0
//a is line segment's size
{
    if (im.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    int n = im.cols/(a-1);
    uchar* pixel = im.data + StringNumber*im.step;

    //l: looking for max/min on the right
    for (int i=a-2; i<n*(a-1); i+=a-1)
    {
        l[i]=pixel[i];
        for (int i1=i-1; i1>i-a+1; i1--)
            l[i1] = flag ? std::max(pixel[i1], l[i1+1]) : std::min(pixel[i1], l[i1+1]);
    }
    l[im.cols-1]=pixel[im.cols-1];
    for (int i1 = im.cols-2; i1>n*(a-1)-1; i1--)
        l[i1] = flag ? std::max(pixel[i1], l[i1+1]) : std::min(pixel[i1], l[i1+1]);

    //r: looking for max/min on the left
    for (int i=0; i<n*(a-1); i+=a-1)
    {
        r[i] = pixel[i];
        for (int i1=i+1; i1<i+a-1; i1++)
            r[i1] = flag ? std::max(pixel[i1], r[i1-1]) : std::min(pixel[i1], r[i1-1]);
    }
    r[n*(a-1)]=pixel[n*(a-1)];
    for (int i1=n*(a-1)+1; i1<im.cols; i1++)
        r[i1] = flag ? std::max(pixel[i1], r[i1-1]) : std::min(pixel[i1], r[i1-1]);
}

cv::Mat VanHerk1dm (const cv::Mat& image, const int d, bool flag)
//if max bool = 1, if min bool = 0
//d is line segment's size
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    if (d==1) return image;

    cv::Mat result (image); //mb there is a constructor making an empty image with image.rows and cols?

   for (int i=0; i<image.rows; i++)
   {
        std::vector<uchar>
             left (image.cols, 0),
             right (image.cols, 0);

        VanHerkInit(image, flag, i, d, left, right);

        uchar* pixel = result.data + i*result.step;
        for (int j=0; j<result.cols; j++)
        {
            auto op1 = left[std::max(0, j-d/2)];
            auto op2 = right[std::min(result.cols-1, j+d/2)];
            pixel[j] = flag ? std::max(op1, op2) : std::min(op1, op2);
        }
   }
   return result;
}

cv::Mat transpon (const cv::Mat& image)
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    cv::Mat result(image.cols, image.rows, cv::IMREAD_GRAYSCALE);
    for (int i=0; i<result.rows; i++) {
        const auto line = result.data + i * result.step;
        for (int j=0; j<result.cols; j++)
            line[j] = (image.data+j*image.step)[i];
    }
    return result;
}

cv::Mat dilatation(const cv::Mat& image, int width, int height)
//width*height is a rectangle (morphology element) size
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    if (!width%2) width++;
    if (!height%2) height++;

    cv::Mat result (image);
    result = VanHerk1dm(result, width, 1);
    result = transpon(result);
    result = VanHerk1dm(result, height, 1);
    result = transpon(result);

    return (result);
}

cv::Mat erosion(const cv::Mat& image, int width, int height)
//width*height is a rectangle (morphology element) size
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    if (!width%2) width++;
    if (!height%2) height++;

    cv::Mat result (image);
    result = VanHerk1dm(result, width, 0);
    result = transpon(result);
    result = VanHerk1dm(result, height, 0);
    result = transpon(result);

    return (result);
}
