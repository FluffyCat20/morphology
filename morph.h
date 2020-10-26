#ifndef MORPH_H
#define MORPH_H

cv::Mat dilatation(const cv::Mat& image, const int width, const int height);
cv::Mat erosion(const cv::Mat& image, const int width, const int height);

#endif // MORPH_H
