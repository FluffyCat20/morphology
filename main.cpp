#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include "niblack_algorithm.h"
#include "vec_methods.h"
#include "frame_rectangle.h"
#include "ccl_algorithm.h"
#include "morph.h"

int main(int argc, char *argv[])
{
    int p = 0;
    double average_width = 0, average_length = 0;
    std::vector<std::vector<int>> imgcopy;
    std::vector<std::vector<int>> comps;
    std::vector<int> widths;
    std::vector<int> lengths;
    cv::Mat img = cv::imread( argv[1], cv::IMREAD_GRAYSCALE);

    if (img.empty())
    {
        std::cout << "error" << std::endl;
        return -1;
    }
    if (img.type() != CV_8UC1)
        throw std::runtime_error("Invalid input image");

    cv::Mat res = img.clone();

    niblack_algorithm(res, 40, -0.2);

    res = erosion(res, 9, 7); //numbers must be odd!
    res = dilatation (res, 5, 5);//numbers must be odd!

    cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
    cv::imshow("window", res);
    cv::waitKey(0);

    ccl_algorithm(res, imgcopy);
    create_set_of_components(imgcopy, comps);

    // filter by the number of pixels
     for (int i = 0; i < comps.size(); i++)
        {
            if (comps[i].size() < 201 || comps[i].size() > 5500)
            {
                comps.erase(comps.begin() + i);
                i = i - 1;
            }
        }

     create_frame_parametrs(comps, widths, lengths);
     average_length = average(lengths);
     average_width = average(widths);

    // filter by height and width
     for (int i = 0; (i < widths.size())&&(i<lengths.size()); i++)
        {
            p = 0;
            if (widths[i] > 2.6 * average_width || widths[i] < 0.2 * average_width)
            {
                delete_component(comps, widths, lengths, i);
                i = i - 1;
                p = 1;
            }
            if ((!p)&&((lengths[i] > 1.4 * average_length)|| (lengths[i] < 0.6 * average_length)))
            {
                delete_component(comps, widths, lengths, i);
                i = i - 1;
            }
        }

    // filter by the quantity of white pixels in the rectangle
     for (int i=0; i<comps.size(); i++)
     {
         double white_pixels = 1 - (double)(comps[i].size())/2/(double)(widths[i]*lengths[i]);
         if (white_pixels > 0.5)
         {
             delete_component(comps, widths, lengths, i);
             i -= 1;
         }
     }

    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    framepaint(img, comps);

    cv::imwrite("output.jpg", img);

    cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
    cv::imshow("window", img);
    cv::waitKey(0);

    return 0;
}
