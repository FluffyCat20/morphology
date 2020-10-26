#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>


void makeBlue(cv::Mat img, int i, int j)
{
    if (img.type() != CV_8UC3)
        throw std::runtime_error("Invalid input image");

    for (int k = 1; k < 3; k++)
        img.at<cv::Vec3b>(i, j)[k] = 0;

    img.at<cv::Vec3b>(i, j)[0] = 255;
}

void framepaint(cv::Mat img1, std::vector<std::vector<int>>const& comps)
{
    int max_j = 0, min_j = 0, max_i = 0, min_i = 0;
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
}

void create_frame_parametrs(std::vector<std::vector<int>>& comps, std::vector<int>& widths, std::vector<int>& lengths)
{
    int max_j = 0, min_j = 0, max_i = 0, min_i = 0;

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
}
