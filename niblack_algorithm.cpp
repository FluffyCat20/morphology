#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

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
