#ifndef MULTI_SESSION_UTIL_H
#define MULTI_SESSION_UTIL_H

#include <Eigen/Geometry>
#include <vector>
#include "STDesc.h"

// Fwd declaration
class STDescManager;
struct ConfigSetting;

// Loop info
struct InterSessionLoop
{
    int ref_kf;
    int cur_kf;
    double score;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose; // T_cur_to_ref
    std::vector<std::pair<STDesc, STDesc>> match_pairs;
};

#endif // MULTI_SESSION_UTIL_H