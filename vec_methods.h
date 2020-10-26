#ifndef VEC_METHODS_H
#define VEC_METHODS_H
#include <vector>

double average(std::vector<double>const& vec);

double average(std::vector<int>const& vec);

void delete_component(std::vector<std::vector<int>>& comps, std::vector<int>& widths, std::vector<int>& lengths, int i);

double maximum(std::vector<double>const& vec);

double minimum(std::vector<double>const& vec);

double maximum(std::vector<int>const& vec);

double minimum(std::vector<int>const& vec);

#endif // VEC_METHODS_H
