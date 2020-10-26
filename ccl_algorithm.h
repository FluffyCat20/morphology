#ifndef CCL_ALGORITHM_H
#define CCL_ALGORITHM_H

void ccl_algorithm(cv::Mat img, std::vector<std::vector<int>>& imgcopy);
void create_set_of_components(std::vector<std::vector<int>>& imgcopy, std::vector<std::vector<int>>& comps);

#endif // CCL_ALGORITHM_H
