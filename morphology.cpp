#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

void niblack_algorithm(cv::Mat &im, const int r, const double k)
{
    double nAverage, average, sqrnDeviation, threshold;
    std::vector<std::vector<double>> brightness (im.rows, std::vector <double>(im.cols, 0));
    uchar* pixel = im.data;
    for (int i=0; i<im.rows; i++)
    {
        for (int j=0; j<im.cols; j++)
            brightness[i][j] = pixel[j];
        pixel += im.step;
    }
    //(i, 0) first
    for (int i = 0; i < im.rows; i++)
    {
        int n = (std::min(im.rows, i + r + 1) - std::max(0, i - r)) * (r + 1);
        average = nAverage = sqrnDeviation = 0;
        for (int i1 = std::max(0, i - r); i1 < std::min(im.rows, i + r + 1); i1++)
            for (int j1 = 0; j1 < r + 1; j1++)
                nAverage += brightness[i1][j1];
        average = nAverage / n;
        for (int i1 = std::max(0, i - r); i1 < std::min(im.rows, i + r + 1); i1++)
            for (int j1 = 0; j1 < r + 1; j1++)
            {
                double b = brightness[i1][j1];
                sqrnDeviation += (b - average) * (b - average);
            }

        threshold = average + k * sqrt(sqrnDeviation / n);
        //if (sqrnDeviation / n < 256*256/50) threshold += 10*k * sqrt(sqrnDeviation / n);
        if (brightness[i][0] < threshold) (im.data+i*im.step)[0]=0; else (im.data+i*im.step)[0]=255;

        for (int j = 1; j < im.cols; j++)
        {
            n = (std::min(im.cols, j + r) - std::max(0, j - r)) * (std::min(im.rows, i + r) - std::max(0, i - r));
            for (int k1 = i - r; k1 < i + r + 1; k1++)
            {
                double b1 = 0;
                if (j - r - 1 >= 0 && k1 >= 0 && k1 < im.rows) b1 = brightness[k1][j - r - 1];
                double b2 = 0;
                if (j + r < im.cols && k1 >= 0 && k1 < im.rows) b2 = brightness[k1][j + r];
                nAverage -= b1;
                nAverage += b2;
            }
            average = nAverage / n;
            for (int k1 = i - r; k1 < i + r + 1; k1++)
            {
                double b1 = average;
                if (j - r - 1 >= 0 && k1 >= 0 && k1 < im.rows) b1 = brightness[k1][j - r - 1];
                double b2 = average;
                if (j + r < im.cols && k1 >= 0 && k1 < im.rows) b2 = brightness[k1][j + r];
                sqrnDeviation -= (b1 - average) * (b1 - average);
                sqrnDeviation += (b2 - average) * (b2 - average);
            }

            if (sqrnDeviation / n < 256 * 256 / 4) sqrnDeviation *= 2;//!!
            threshold = average + k * sqrt(sqrnDeviation / n);
            //if (sqrnDeviation / n < 256*256/64) threshold += 10*k*sqrt(sqrnDeviation/n);
            if (brightness[i][j] < threshold) (im.data+i*im.step)[j]=0; else (im.data+i*im.step)[j]=255;
        }
    }
}


void VanHerkInit(cv::Mat& im, bool flag, const int StringNumber, const int a, std::vector<uchar> &l, std::vector<uchar> &r)
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
            if (flag) l[i1] = std::max(pixel[i1], l[i1+1]);
                else l[i1] = std::min(pixel[i1], l[i1+1]);
    }
    l[im.cols-1]=pixel[im.cols-1];
    for (int i1 = im.cols-2; i1>n*(a-1)-1; i1--)
        if (flag) l[i1] = std::max(pixel[i1], l[i1+1]);
            else l[i1] = std::min(pixel[i1], l[i1+1]);

    //r: looking for max/min on the left
    for (int i=0; i<n*(a-1); i+=a-1)
    {
        r[i] = pixel[i];
        for (int i1=i+1; i1<i+a-1; i1++)
            if (flag) r[i1] = std::max(pixel[i1], r[i1-1]);
                else r[i1] = std::min(pixel[i1], r[i1-1]);
    }
    r[n*(a-1)]=pixel[n*(a-1)];
    for (int i1=n*(a-1)+1; i1<im.cols; i1++)
        if (flag) r[i1] = std::max(pixel[i1], r[i1-1]);
            else r[i1] = std::min(pixel[i1], r[i1-1]);
}

cv::Mat VanHerk1dm (cv::Mat& image, const int d, bool flag)
//if max bool = 1, if min bool = 0
//d is line segment's size
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    cv::Mat result(image); //mb there is a constructor making an empty image with image.rows and cols?

   for (int i=0; i<image.rows; i++)
   {
        std::vector<uchar>
             left (image.cols, 0),
             right (image.cols, 0);

        VanHerkInit(image, flag, i, d, left, right);

        uchar* pixel = result.data + i*result.step;
        for (int j=0; j<result.cols; j++)
        {
            if (flag) pixel[j] = std::max(left[std::max(0, j-d/2)], right[std::min(result.cols-1, j+d/2)]);
                else pixel[j] = std::min(left[std::max(0, j-d/2)], right[std::min(result.cols-1, j+d/2)]);
        }
   }
   return result;
}

cv::Mat transpon (cv::Mat& image)
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    cv::Mat result(image.cols, image.rows, cv::IMREAD_GRAYSCALE);
    for (int i=0; i<result.rows; i++)
        for (int j=0; j<result.cols; j++)
            (result.data+i*result.step)[j] = (image.data+j*image.step)[i];
    return result;
}

cv::Mat dilatation(cv::Mat &image, const int width, const int height)
//width*height is a rectangle (morphology element) size
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    cv::Mat result(image);
    result = VanHerk1dm(result, width, 1);
    result = transpon(result);
    result = VanHerk1dm(result, height, 1);
    result = transpon(result);

    return (result);
}

cv::Mat erosion(cv::Mat &image, const int width, const int height)
//width*height is a rectangle (morphology element) size
{
    if (image.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    cv::Mat result(image);
    result = VanHerk1dm(result, width, 0);
    result = transpon(result);
    result = VanHerk1dm(result, height, 0);
    result = transpon(result);

    return (result);
}

int find_(int X, std::vector<int> const& linked)
{
    int j = X;
    while (linked[j-1] != 0)
        j = linked[j-1];
    return j;

}

void union_(int X, int Y, std::vector<int>& linked)
{
    int j = find_(X, linked);
    int k = find_(Y, linked);
    if (j != k)
        linked[k-1] = j;
}

void makeBlue(cv::Mat img1, int i, int j)
{
    if(img1.type() != CV_8UC3)
        throw std::runtime_error("Invalid input image");
    for (int k = 1; k < 3; k++)
        img1.at<cv::Vec3b>(i, j)[k] = 0;
    img1.at<cv::Vec3b>(i, j)[0] = 255;
}

cv::Mat frame (cv::Mat& img)
{
    if(img.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");
    std::cout << "ok" << std::endl;
    std::vector<std::vector<int>> imgcopy;
    std::vector<int> linked;
    std::vector<std::vector<int>> comps;
    std::vector<int> use;
    std::vector<int> widths;
    std::vector<int> lengths;
    std::vector<double> ratio;
    int label = 0, B = 0, C = 0, p = 0, max_j = 0, min_j = 0, max_i = 0, min_i = 0, index = 0, average_width = 0, average_length = 0, help1 = 0, help2 = 0;
    int max_width = 0, min_width = 0, max_length = 0, min_length = 0;
    double  max_dens = 0, min_dens = 0, average_dens = 0;
    double average_ratio = 0, help3 = 0;
    for (int i = 0; i < img.rows; i++)
    {
        imgcopy.push_back(std::vector<int>());

        for (int j = 0; j < img.cols; j++)
            {
            if (img.at<uchar>(i, j) == 255)
            {
                imgcopy[i].push_back(0);
            }
            else
            {
                if (i == 0)
                {
                    if (j == 0)
                    {
                        label +=1;
                        linked.push_back(0);
                        imgcopy[i].push_back(label);
                    }
                    if (j > 0)
                    {
                        if (imgcopy[i][j - 1] != 0)
                        {
                            imgcopy[i].push_back(imgcopy[i][j - 1]);
                        }
                        else
                        {
                            label = label + 1;
                            linked.push_back(0);
                            imgcopy[i].push_back(label);
                        }
                    }
                }
                if ((j == 0)&&(i!=0))
                {
                    if (imgcopy[i - 1][j] != 0)
                    {
                        imgcopy[i].push_back(imgcopy[i - 1][j]);
                    }
                    else
                    {
                        label = label + 1;
                        linked.push_back(0);
                        imgcopy[i].push_back(label);
                     }
                }
                if ((i > 0) && (j > 0))
                {
                    B = imgcopy[i][j - 1];
                    C = imgcopy[i - 1][j];
                    if ((B == 0) && (C == 0))
                    {
                        label = label + 1;
                        linked.push_back(0);
                        imgcopy[i].push_back(label);
                    }
                    if ((B != 0) && (C == 0))
                    {
                        imgcopy[i].push_back(B);
                    }

                    if ((B == 0) && (C != 0))
                    {
                        imgcopy[i].push_back(C);
                    }
                    if ((B != 0) && (C != 0))
                    {
                        if (B == C)
                        {
                            imgcopy[i].push_back(B);
                        }
                        else
                        {
                            if (B > C)
                            {
                                linked[B-1] = C;
                                imgcopy[i].push_back(C);
                                union_(C, B, linked);
                            }
                            else
                            {
                                linked[C-1] = B;
                                imgcopy[i].push_back(B);
                                union_(B, C, linked);
                            }
                        }
                    }
                }
            }
            }
    }
    for (int i = 0; i < imgcopy.size(); i++)
    {
        for (int j = 0; j < imgcopy[i].size(); j++)
        {
            if (img.at<uchar>(i, j) == 0)
                imgcopy[i][j] = find_(imgcopy[i][j], linked);
        }
    }
    //end of connected-component label algorithm

    for (int i = 0; i < imgcopy.size(); i++)
    {
        for (int j = 0; j < imgcopy[i].size(); j++)
        {
            p = 0;
            if (imgcopy[i][j] != 0)
            {
                for (int m = 0; m < comps.size(); m++)
                    if (comps[m][0] == imgcopy[i][j])
                    {
                        comps[m].push_back(i);
                        comps[m].push_back(j);
                        p = 1;
                    }
                if (p == 0)
                {
                    use.push_back(imgcopy[i][j]);
                    comps.push_back(use);
                    use.pop_back();
                }
            }
        }
    }

    // 1. IF THE CONNECTIVITY COMPONENT HAS LESS THAN 20 PIXELS, WE DON'T CONSIDER IT
        for (int i = 0; i < comps.size(); i++)
        {
            if (comps[i].size() < 41)
            {

                comps.erase(comps.begin() + i);
                std::vector<std::vector<int>>(comps).swap(comps);
                i = i - 1;
            }

        }

    // calculating the  average width and height of connectivity components
        for (int i = 0; i < comps.size(); i++)
        {

            max_j = comps[i][2];
            min_j = comps[i][2];
            min_i = comps[i][1];
            max_i = comps[i][1];
            for (int j = 2; j < comps[i].size(); j +=2)
            {

                if (comps[i][j] > max_j)
                    max_j = comps[i][j];
                if (comps[i][j] < min_j)
                    min_j = comps[i][j];
                if (comps[i][j-1] > max_i)
                    max_i = comps[i][j-1];
                if (comps[i][j-1] < min_i)
                    min_i = comps[i][j-1];

            }
            widths.push_back(max_j - min_j + 1);
            lengths.push_back(max_i - min_i + 1);

        }
        for (int i = 0; i < widths.size(); i++)
        {
            ratio.push_back(((double)widths[i])/(double)lengths[i]);
        }

        for (int i = 0; i < widths.size(); i++)
        {
            help1 = help1 + widths[i];
            help2 = help2 + lengths[i];
            help3 = help3 + ratio[i];
        }
        average_length = help1 /(widths.size());
        average_width = help2 / (lengths.size());
        average_ratio = help3 / (ratio.size());

        //painting framing rectangles
            cv::Mat img1(img);
            cv::cvtColor(img1, img1, cv::COLOR_GRAY2BGR);
            for (int i = 0; i < comps.size(); i++)
            {

                max_j = comps[i][2];
                min_j = comps[i][2];
                min_i = comps[i][1];
                max_i = comps[i][1];
                for (int j = 2; j < comps[i].size(); j += 2)
                {

                    if (comps[i][j] > max_j)
                        max_j = comps[i][j];
                    if (comps[i][j] < min_j)
                        min_j = comps[i][j];
                    if (comps[i][j - 1] > max_i)
                        max_i = comps[i][j - 1];
                    if (comps[i][j - 1] < min_i)
                        min_i = comps[i][j - 1];

                }
                for (int index1 = min_i; index1 <= max_i; index1++)
                {
                    makeBlue(img1, index1, min_j);
                    makeBlue(img1, index1, max_j);
                }
                for (int index1 = min_j; index1 <= max_j; index1++)
                {
                    makeBlue(img1, min_i, index1);
                    makeBlue(img1, max_i, index1);
                }

            }

    return img1;
}

int main(int argc, char *argv[])
{
    //int d1=13, d2=3; //must be odd!
    cv::Mat img = cv::imread( argv[1], cv::IMREAD_GRAYSCALE);

    if (img.empty())
    {
        std::cout << "error" << std::endl;
        return -1;
    }
    if (img.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    cv::Mat res (img);

    niblack_algorithm(res, 40, -0.2);
    res = erosion(res, 9, 9); //numbers must be odd!
    res = dilatation (res, 5, 3);//numbers must be odd!
    //res = erosion(res, 5, 7); //numbers must be odd!

    std::cout << "before frame" << std::endl;
    res = frame(res);

    cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
    cv::imshow("window", res);
    cv::waitKey(0);

    cv::destroyWindow("window");

    return 0;
}
