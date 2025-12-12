#ifndef MULTI_SESSION_UTIL_H
#define MULTI_SESSION_UTIL_H

#include <Eigen/Geometry>
#include <vector>
#include "STDesc.h"
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Fwd declaration
class STDescManager;
struct ConfigSetting;
typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;
// Loop info
struct InterSessionLoop
{
    int ref_kf;
    int cur_kf;
    double score;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose; // T_cur_to_ref
    std::vector<std::pair<STDesc, STDesc>> match_pairs;
};

// 根据当前 session 的 stds，与上一 session 的 db 做 inter-session loop 搜索
bool detectInterSessionLoopForKeyframe(
    int cur_kf,
    const std::vector<STDesc> &cur_stds,
    const std::shared_ptr<STDescManager> &std_manager_ref,
    const ConfigSetting &config_setting,
    InterSessionLoop &out_loop);

bool loadSessionAndBuildSTD(
    std::vector<Eigen::Affine3d> &keyframe_poses,
    std::vector<PointCloud::Ptr> &keyframe_clouds,
    const ConfigSetting &config_setting,
    int session_id,
    const std::shared_ptr<STDescManager> &std_manager_ref);

bool detectInterSessionLoops(
    const std::vector<PointCloud::Ptr> &keyframe_clouds_cur,
    const std::shared_ptr<STDescManager> &std_manager_ref,
    const ConfigSetting &config_setting,
    std::vector<InterSessionLoop> &inter_loops);

Eigen::Affine3d estimateSessionTransform(
    const std::vector<InterSessionLoop> &inter_loops,
    bool yaw_only);
    
void visualizeMultiSessionLoop(
    const ros::Publisher &publisher,
    const Eigen::Affine3d &pose_cur,
    const Eigen::Affine3d &pose_ref);
#endif // MULTI_SESSION_UTIL_H