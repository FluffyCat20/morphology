#ifndef FRAME_RECTANGLE_H
#define FRAME_RECTANGLE_H

void makeBlue(cv::Mat img, int i, int j);

void framepaint(cv::Mat img1, std::vector<std::vector<int>>const& comps);

void create_frame_parametrs(std::vector<std::vector<int>>& comps, std::vector<int>& widths, std::vector<int>& lengths);

#endif // FRAME_RECTANGLE_H
