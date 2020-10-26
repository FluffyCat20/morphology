#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

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

void ccl_algorithm(const cv::Mat img, std::vector<std::vector<int>>& imgcopy)
{
std::vector<int> linked;
int label = 0, B = 0, C = 0;

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
                        label = label + 1;
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
}

void create_set_of_components(std::vector<std::vector<int>>& imgcopy, std::vector<std::vector<int>>& comps)
{
    std::vector<int> use;
    int p = 0;

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
}
