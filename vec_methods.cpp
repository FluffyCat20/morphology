#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

double average(std::vector<double>const& vec)
{
    if (vec.size() == 0)
        throw std::runtime_error("Search for the average in an empty array");

    double help1 = 0;
    for (int i = 0; i < vec.size(); i++)
            help1 = help1 + vec[i];

    return (help1/vec.size());
}

double average(std::vector<int>const& vec)
{
    if (vec.size() == 0)
        throw std::runtime_error("Search for the average in an empty array");

    int help1 = 0;
    for (int i = 0; i < vec.size(); i++)
            help1 = help1 + vec[i];

    return ((double)help1/(double)vec.size());
}

void delete_component(std::vector<std::vector<int>>& comps, std::vector<int>& widths, std::vector<int>& lengths, int i)
{
     comps.erase(comps.begin() + i);

     widths.erase(widths.begin() + i);

     lengths.erase(lengths.begin() + i);
}

double maximum(std::vector<double>const& vec)
{
   return vec[std::max_element(vec.begin(), vec.end()) - vec.begin()];
}

double minimum(std::vector<double>const& vec)
{
   return vec[std::min_element(vec.begin(), vec.end()) - vec.begin()];
}

double maximum(std::vector<int>const& vec)
{
   return vec[std::max_element(vec.begin(), vec.end()) - vec.begin()];
}

double minimum(std::vector<int>const& vec)
{
   return vec[std::min_element(vec.begin(), vec.end()) - vec.begin()];
}
