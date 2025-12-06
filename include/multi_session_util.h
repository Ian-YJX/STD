#include <vector>
#include <iostream>
#include "STDesc.h"

struct InterSessionLoop
{
  int ref_kf;   // reference session 的关键帧 index (session 0)
  int cur_kf;   // current  session 的关键帧 index (session 1)

  double score; // SearchLoop 输出的匹配得分

  std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose; // triangle solver 得到的相对位姿
  std::vector<std::pair<STDesc, STDesc>> match_pairs;        // 匹配上的描述子对（用于可视化/调试）
};