// offline_multi_session.cpp
// 离线多会话拼地图 + 可视化

#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include "include/STDesc.h"
#include "include/multi_session_util.h"

// ----------------------------------
// 一些 typedef
// ----------------------------------
using PointType = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointType>;

// ----------------------------------
// 功能3：根据回环估计会话级变换 T_{W1->W0}
//
// 关键点：用到了“第二个 T 矩阵”（回环相对位姿）
//   - loop.relative_pose: T_F0_F1   （把 session1 的帧坐标 F1 变换到 session0 的帧坐标 F0）
//   - keyframe_poses_ref[k0] : T_W0_F0
//   - keyframe_poses_cur[k1] : T_W1_F1
//
// 物理点 P 在两个会话满足：
//   P_W0 = T_W0_F0 * T_F0_F1 * P_F1
//   P_W1 = T_W1_F1 * P_F1
//   P_W0 = T_W1_to_W0 * P_W1
//
// 推出：
//
//   T_W1_to_W0 * T_W1_F1 = T_W0_F0 * T_F0_F1
// => T_W1_to_W0 = T_W0_F0 * T_F0_F1 * (T_W1_F1)^{-1}
//
// 对每条回环求一个 T_i，然后对 {T_i} 做加权平均（score 为权重）。
// ----------------------------------
Eigen::Affine3d estimateSessionTransformFromLoops(
    const std::vector<Eigen::Affine3d> &keyframe_poses_ref,
    const std::vector<Eigen::Affine3d> &keyframe_poses_cur,
    const std::vector<InterSessionLoop> &inter_loops)
{
    Eigen::Affine3d T_W1_to_W0 = Eigen::Affine3d::Identity();

    if (inter_loops.empty())
    {
        std::cerr << "[estimateSessionTransformFromLoops] No inter-session loops, "
                  << "return Identity." << std::endl;
        return T_W1_to_W0;
    }

    // ---- 收集每条 loop 给出的 T_i，并把旋转“投影”为纯 yaw ----
    std::vector<double> yaws;           // 每条 loop 的 yaw（绕 Z）
    std::vector<Eigen::Vector3d> trans; // 每条 loop 的平移
    std::vector<double> weights;        // 权重：用 loop.score

    yaws.reserve(inter_loops.size());
    trans.reserve(inter_loops.size());
    weights.reserve(inter_loops.size());

    for (const auto &loop : inter_loops)
    {
        int k0 = loop.ref_kf;
        int k1 = loop.cur_kf;

        if (k0 < 0 || k0 >= (int)keyframe_poses_ref.size() ||
            k1 < 0 || k1 >= (int)keyframe_poses_cur.size())
        {
            std::cerr << "[estimateSessionTransformFromLoops] Skip loop with invalid "
                      << "index: ref_kf=" << k0 << ", cur_kf=" << k1 << std::endl;
            continue;
        }

        // T_i = T_ref * T_cur^{-1}  ≈  T_W1_to_W0
        Eigen::Affine3d T_ref = keyframe_poses_ref[k0];
        Eigen::Affine3d T_cur = keyframe_poses_cur[k1];
        Eigen::Affine3d T_i = T_ref * T_cur.inverse();

        Eigen::Matrix3d R_i = T_i.rotation();
        Eigen::Vector3d t_i = T_i.translation();

        // 从 R_i 提取 yaw（绕 Z 轴），对应 ROS 约定：X前 Y左 Z上
        double yaw_i = std::atan2(R_i(1, 0), R_i(0, 0));

        double w_i = std::max(loop.score, 1e-3); // 防止 0 权重

        yaws.push_back(yaw_i);
        trans.push_back(t_i);
        weights.push_back(w_i);
    }

    if (yaws.empty())
    {
        std::cerr << "[estimateSessionTransformFromLoops] No valid loops after "
                  << "filtering, return Identity." << std::endl;
        return T_W1_to_W0;
    }

    // ---- 1) yaw 的加权圆均值（只保留绕 Z 的旋转）----
    double w_sum = 0.0;
    double cos_sum = 0.0;
    double sin_sum = 0.0;

    for (size_t i = 0; i < yaws.size(); ++i)
    {
        double w = weights[i];
        double yaw = yaws[i];

        cos_sum += w * std::cos(yaw);
        sin_sum += w * std::sin(yaw);
        w_sum += w;
    }

    if (w_sum <= 0.0)
    {
        std::cerr << "[estimateSessionTransformFromLoops] Sum of weights is zero, "
                  << "return Identity." << std::endl;
        return T_W1_to_W0;
    }

    double yaw_avg = std::atan2(sin_sum, cos_sum);
    Eigen::AngleAxisd aa_yaw(yaw_avg, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d R_avg = aa_yaw.toRotationMatrix(); // 只有 yaw，无 roll/pitch

    // ---- 2) 平移的加权平均（这里保留 x,y,z 三个分量）----
    Eigen::Vector3d t_sum(0, 0, 0);
    for (size_t i = 0; i < trans.size(); ++i)
    {
        t_sum += weights[i] * trans[i];
    }
    Eigen::Vector3d t_avg = t_sum / w_sum;

    // 如果你想更“平面”，也可以在这里强制 z=某个值，例如：
    // t_avg.z() = 0.0;  // 或者 t_avg.z() 设成第一条 loop 的 z

    T_W1_to_W0 = Eigen::Affine3d::Identity();
    T_W1_to_W0.linear() = R_avg;
    T_W1_to_W0.translation() = t_avg;

    // ---- 打印一下结果，方便你在终端里看 ----
    std::cout << "[estimateSessionTransformFromLoops] use " << yaws.size()
              << " loops, final T_W1_to_W0 (yaw-only rotation):\n";
    std::cout << "  yaw (deg) = " << yaw_avg * 180.0 / M_PI << std::endl;
    std::cout << "  t = " << t_avg.transpose() << std::endl;
    std::cout << T_W1_to_W0.matrix() << std::endl;

    return T_W1_to_W0;
}

Eigen::Affine3d estimateSessionTransformFromLoops_STD(
    const std::vector<Eigen::Affine3d> &keyframe_poses_ref,
    const std::vector<Eigen::Affine3d> &keyframe_poses_cur,
    const std::vector<InterSessionLoop> &inter_loops)
{
    Eigen::Affine3d T_W1_to_W0 = Eigen::Affine3d::Identity();

    if (inter_loops.empty())
    {
        std::cerr << "[estimateSessionTransformFromLoops_STD] No inter-session loops, return Identity."
                  << std::endl;
        return T_W1_to_W0;
    }

    // ----------------------------------------------------------
    // 1) 先按 score 过滤一遍，留下“质量高”的 loops
    //    你现在 icp_threshold = 0.3，这里稍微高一点 0.35
    //    不想太激进的话可以改成 0.33 ~ 0.36 自己试
    // ----------------------------------------------------------
    const double SCORE_MIN = 0.35;

    std::vector<const InterSessionLoop *> good_loops;
    good_loops.reserve(inter_loops.size());

    for (const auto &loop : inter_loops)
    {
        if (loop.score >= SCORE_MIN)
        {
            int k0 = loop.ref_kf;
            int k1 = loop.cur_kf;
            if (k0 < 0 || k0 >= (int)keyframe_poses_ref.size() ||
                k1 < 0 || k1 >= (int)keyframe_poses_cur.size())
            {
                continue;
            }
            good_loops.push_back(&loop);
        }
    }

    // 若过滤完太少了，就退回使用全部 loops（防止极端情况）
    if (good_loops.size() < 5)
    {
        std::cout << "[estimateSessionTransformFromLoops_STD] good_loops < 5, fallback to all loops."
                  << std::endl;
        good_loops.clear();
        for (const auto &loop : inter_loops)
        {
            int k0 = loop.ref_kf;
            int k1 = loop.cur_kf;
            if (k0 < 0 || k0 >= (int)keyframe_poses_ref.size() ||
                k1 < 0 || k1 >= (int)keyframe_poses_cur.size())
            {
                continue;
            }
            good_loops.push_back(&loop);
        }
    }

    if (good_loops.empty())
    {
        std::cerr << "[estimateSessionTransformFromLoops_STD] No valid loops after filtering, return Identity."
                  << std::endl;
        return T_W1_to_W0;
    }

    std::cout << "[estimateSessionTransformFromLoops_STD] use "
              << good_loops.size() << " loops for yaw + translation." << std::endl;

    // ----------------------------------------------------------
    // 2) 用 STD relative_pose 的旋转估计平均 yaw
    //    yaw_i = atan2(R_rel(1,0), R_rel(0,0))
    // ----------------------------------------------------------
    double w_sum_yaw = 0.0;
    Eigen::Vector2d yaw_vec(0.0, 0.0); // 加权 (cos, sin)

    for (const auto *ploop : good_loops)
    {
        const auto &loop = *ploop;
        const Eigen::Matrix3d &R_rel = loop.relative_pose.second;

        double yaw_i = std::atan2(R_rel(1, 0), R_rel(0, 0));
        double w = std::max(loop.score, 1e-3);

        yaw_vec += w * Eigen::Vector2d(std::cos(yaw_i), std::sin(yaw_i));
        w_sum_yaw += w;
    }

    double yaw = 0.0;
    if (w_sum_yaw > 0.0)
    {
        yaw = std::atan2(yaw_vec.y(), yaw_vec.x());
    }

    std::cout << "[estimateSessionTransformFromLoops_STD] yaw (deg) = "
              << yaw * 180.0 / M_PI << std::endl;

    Eigen::Matrix3d R_yaw = Eigen::Matrix3d::Identity();
    R_yaw(0, 0) = std::cos(yaw);
    R_yaw(0, 1) = -std::sin(yaw);
    R_yaw(1, 0) = std::sin(yaw);
    R_yaw(1, 1) = std::cos(yaw);
    // R_yaw(2,2) = 1;

    // ----------------------------------------------------------
    // 3) 固定 R_yaw，用 keyframe 平移做加权平均：
    //    p_ref_i  (W0)
    //    p_cur_i  (W1)
    //    p_cur_i' = R_yaw * p_cur_i
    //    理想关系: p_ref_i ≈ p_cur_i' + t  =>  t_i = p_ref_i - p_cur_i'
    // ----------------------------------------------------------
    Eigen::Vector2d t_xy(0.0, 0.0);
    double t_z = 0.0;
    double w_sum_t = 0.0;

    for (const auto *ploop : good_loops)
    {
        const auto &loop = *ploop;
        int k0 = loop.ref_kf;
        int k1 = loop.cur_kf;

        Eigen::Vector3d p_ref = keyframe_poses_ref[k0].translation();
        Eigen::Vector3d p_cur = keyframe_poses_cur[k1].translation();

        Eigen::Vector3d p_cur_rot = R_yaw * p_cur;

        Eigen::Vector2d delta_xy = p_ref.head<2>() - p_cur_rot.head<2>();
        double delta_z = p_ref.z() - p_cur_rot.z();

        double w = std::max(loop.score, 1e-3);

        t_xy += w * delta_xy;
        t_z += w * delta_z;
        w_sum_t += w;
    }

    if (w_sum_t <= 0.0)
    {
        std::cerr << "[estimateSessionTransformFromLoops_STD] Sum of translation weights is zero, return Identity."
                  << std::endl;
        return T_W1_to_W0;
    }

    t_xy /= w_sum_t;
    t_z /= w_sum_t;

    Eigen::Vector3d t(t_xy.x(), t_xy.y(), t_z);

    // ----------------------------------------------------------
    // 4) 组装最终 T_W1_to_W0
    // ----------------------------------------------------------
    T_W1_to_W0 = Eigen::Affine3d::Identity();
    T_W1_to_W0.linear() = R_yaw;
    T_W1_to_W0.translation() = t;

    std::cout << "[estimateSessionTransformFromLoops_STD] final T_W1_to_W0:\n"
              << T_W1_to_W0.matrix() << std::endl;

    return T_W1_to_W0;
}

// ----------------------------------
// 功能4（上）：构建合并后的点云和两条轨迹
//
// 假设：保存下来的 keyframe 点云已经在各自会话的世界坐标系下
//   * session0：直接使用其点云和位姿
//   * session1：先用 T_W1_to_W0 把点云 & 位姿变到 W0 下
// ----------------------------------
void buildMergedMapAndPaths(
    const std::vector<Eigen::Affine3d> &keyframe_poses_ref,
    const std::vector<PointCloud::Ptr> &keyframe_clouds_ref,
    const std::vector<Eigen::Affine3d> &keyframe_poses_cur,
    const std::vector<PointCloud::Ptr> &keyframe_clouds_cur,
    const Eigen::Affine3d &T_W1_to_W0, // 会话级变换：W1 -> W0
    PointCloud::Ptr &merged_map,
    nav_msgs::Path &path_ref,
    nav_msgs::Path &path_cur,
    const std::string &frame_id = "camera_init")
{
    merged_map.reset(new PointCloud);
    path_ref.poses.clear();
    path_cur.poses.clear();

    path_ref.header.frame_id = frame_id;
    path_cur.header.frame_id = frame_id;

    // 1. session0：轨迹 + 点云（点云假设已在 W0）
    for (size_t k = 0; k < keyframe_clouds_ref.size(); ++k)
    {
        // 轨迹
        geometry_msgs::PoseStamped ps;
        ps.header.frame_id = frame_id;
        ps.pose.position.x = keyframe_poses_ref[k].translation().x();
        ps.pose.position.y = keyframe_poses_ref[k].translation().y();
        ps.pose.position.z = keyframe_poses_ref[k].translation().z();
        Eigen::Quaterniond q_ref(keyframe_poses_ref[k].rotation());
        ps.pose.orientation.x = q_ref.x();
        ps.pose.orientation.y = q_ref.y();
        ps.pose.orientation.z = q_ref.z();
        ps.pose.orientation.w = q_ref.w();
        path_ref.poses.push_back(ps);

        // 点云
        PointCloud tmp;
        pcl::transformPointCloud(*keyframe_clouds_ref[k], tmp,
                                 Eigen::Affine3d::Identity());
        *merged_map += tmp;
    }

    // 2. session1：先用 T_W1_to_W0 把位姿 & 点云变换到 W0
    for (size_t k = 0; k < keyframe_clouds_cur.size(); ++k)
    {
        Eigen::Affine3d T_global = T_W1_to_W0 * keyframe_poses_cur[k];

        // 轨迹
        geometry_msgs::PoseStamped ps;
        ps.header.frame_id = frame_id;
        ps.pose.position.x = T_global.translation().x();
        ps.pose.position.y = T_global.translation().y();
        ps.pose.position.z = T_global.translation().z();
        Eigen::Quaterniond q_global(T_global.rotation());
        ps.pose.orientation.x = q_global.x();
        ps.pose.orientation.y = q_global.y();
        ps.pose.orientation.z = q_global.z();
        ps.pose.orientation.w = q_global.w();
        path_cur.poses.push_back(ps);

        // 点云：逐点从 W1 映射到 W0
        PointCloud tmp;
        pcl::transformPointCloud(*keyframe_clouds_cur[k], tmp, T_W1_to_W0);
        *merged_map += tmp;
    }
}

// ----------------------------------
// 功能4（下）：把跨会话 loop 画成一堆连线，方便在 RViz 里看
// ----------------------------------
void buildInterSessionLoopMarkers(
    const std::vector<Eigen::Affine3d> &keyframe_poses_ref,
    const std::vector<Eigen::Affine3d> &keyframe_poses_cur,
    const Eigen::Affine3d &T_W1_to_W0,
    const std::vector<InterSessionLoop> &inter_loops,
    visualization_msgs::MarkerArray &marker_array,
    const std::string &frame_id = "camera_init")
{
    marker_array.markers.clear();
    if (inter_loops.empty())
        return;

    visualization_msgs::Marker marker_edge;
    marker_edge.header.frame_id = frame_id;
    marker_edge.ns = "inter_session_loops";
    marker_edge.id = 0;
    marker_edge.type = visualization_msgs::Marker::LINE_LIST;
    marker_edge.action = visualization_msgs::Marker::ADD;
    marker_edge.scale.x = 0.5;
    marker_edge.color.r = 1.0;
    marker_edge.color.g = 0.0;
    marker_edge.color.b = 0.0;
    marker_edge.color.a = 1.0;
    marker_edge.pose.orientation.w = 1.0;

    for (const auto &loop : inter_loops)
    {
        int k0 = loop.ref_kf;
        int k1 = loop.cur_kf;

        if (k0 < 0 || k0 >= (int)keyframe_poses_ref.size() ||
            k1 < 0 || k1 >= (int)keyframe_poses_cur.size())
            continue;

        Eigen::Vector3d p_ref = keyframe_poses_ref[k0].translation();
        Eigen::Vector3d p_cur = (T_W1_to_W0 * keyframe_poses_cur[k1]).translation();

        geometry_msgs::Point p0, p1;
        p0.x = p_ref.x();
        p0.y = p_ref.y();
        p0.z = p_ref.z();

        p1.x = p_cur.x();
        p1.y = p_cur.y();
        p1.z = p_cur.z();

        marker_edge.points.push_back(p0);
        marker_edge.points.push_back(p1);
    }

    marker_array.markers.push_back(marker_edge);
}

// 在 offline_multi_session.cpp 末尾添加
void optimizeMultiSessionPoses(
    const std::vector<Eigen::Affine3d> &poses_ref, // 会话0的位姿（已单会话优化）
    const std::vector<Eigen::Affine3d> &poses_cur, // 会话1的位姿（已单会话优化）
    const std::vector<InterSessionLoop> &inter_loops,
    const Eigen::Affine3d &T_W1_to_W0_initial, // 初始会话间变换
    std::vector<Eigen::Affine3d> &optimized_poses_ref,
    std::vector<Eigen::Affine3d> &optimized_poses_cur)
{
    // 1. 定义噪声模型 - 关键：利用已优化的单会话结果
    // 由于位姿已通过单会话优化，里程计噪声较小
    auto odom_noise = gtsam::noiseModel::Diagonal::Sigmas(
        (gtsam::Vector(6) << 0.01, 0.01, 0.00000005, 0.0000005, 0.00000005, 0.05).finished());

    // 回环噪声：基于匹配得分动态调整，得分越高约束越强
    auto loop_noise_model = [](double score) -> gtsam::SharedNoiseModel
    {
        double weight = 1.0 / (0.1 + 0.9 * (1.0 - score));
        return gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 0.1 * weight, 0.1 * weight, 0.1 * weight,
             0.05 * weight, 0.05 * weight, 0.05 * weight)
                .finished());
    };

    // 2. 创建因子图和初始值
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values initial;

    // 3. 添加会话0的位姿节点（直接使用已优化位姿）
    for (size_t i = 0; i < poses_ref.size(); i++)
    {
        // gtsam::Pose3 pose(gtsam::Pose3::MatrixType(poses_ref[i].matrix()));
        gtsam::Pose3 pose(poses_ref[i].matrix());
        initial.insert(i, pose);

        // 添加相邻关键帧之间的里程计约束
        if (i > 0)
        {
            gtsam::Pose3 prev_pose(poses_ref[i - 1].matrix());
            gtsam::Pose3 curr_pose(poses_ref[i].matrix());
            gtsam::Pose3 odom = prev_pose.between(curr_pose);
            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1, i, odom, odom_noise));
        }
    }

    // 4. 添加会话1的位姿节点（转换到会话0坐标系下）
    const size_t offset = poses_ref.size();
    for (size_t j = 0; j < poses_cur.size(); j++)
    {
        // 将会话1的位姿转换到会话0坐标系下作为初始值
        Eigen::Matrix4d global_mat = (T_W1_to_W0_initial * poses_cur[j]).matrix();
        gtsam::Pose3 global_pose(global_mat);
        initial.insert(offset + j, global_pose);

        // 添加相邻关键帧之间的里程计约束
        if (j > 0)
        {
            gtsam::Pose3 prev_pose(poses_cur[j - 1].matrix());
            gtsam::Pose3 curr_pose(poses_cur[j].matrix());
            gtsam::Pose3 odom = prev_pose.between(curr_pose);
            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(offset + j - 1, offset + j, odom, odom_noise));
        }
    }

    // 5. 添加跨会话回环约束（关键！）
    for (const auto &loop : inter_loops)
    {
        // 将回环相对位姿转换为gtsam::Pose3
        gtsam::Rot3 R(loop.relative_pose.second);
        gtsam::Point3 t(loop.relative_pose.first);
        gtsam::Pose3 loop_pose(R, t);

        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            loop.ref_kf,          // 会话0关键帧索引
            offset + loop.cur_kf, // 会话1关键帧索引
            loop_pose,
            loop_noise_model(loop.score)));
    }

    // 在添加所有约束后，添加先验约束 - 关键修复
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
        gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    // 固定会话0的第一个位姿作为全局参考系
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(), priorNoise));

    // 6. 执行增量优化
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    gtsam::ISAM2 isam(parameters);

    isam.update(graph, initial);
    gtsam::Values result = isam.calculateEstimate();

    // 7. 提取优化结果
    optimized_poses_ref = poses_ref; // 初始化为原值（安全）
    optimized_poses_cur = poses_cur;

    for (size_t i = 0; i < poses_ref.size(); i++)
    {
        optimized_poses_ref[i] = Eigen::Affine3d(result.at<gtsam::Pose3>(i).matrix());
    }
    for (size_t j = 0; j < poses_cur.size(); j++)
    {
        optimized_poses_cur[j] = Eigen::Affine3d(result.at<gtsam::Pose3>(offset + j).matrix());
    }
}

// ----------------------------------
// （工具函数：目前不在 main 里用，但保留无妨）
// ----------------------------------
void update_poses(const gtsam::Values &estimates,
                  std::vector<Eigen::Affine3d> &poses)
{
    assert(estimates.size() == poses.size());
    poses.clear();

    for (size_t i = 0; i < estimates.size(); ++i)
    {
        auto est = estimates.at<gtsam::Pose3>(i);
        Eigen::Affine3d est_affine3d(est.matrix());
        poses.push_back(est_affine3d);
    }
}

// ----------------------------------
// 主函数
// ----------------------------------
int main(int argc, char **argv)
{
    ros::init(argc, argv, "offline_multi_session");
    ros::NodeHandle nh;

    ConfigSetting config_setting;
    read_parameters(nh, config_setting);

    // Publisher：统一 /multi_session/* 这些 topic
    ros::Publisher pubMergedCloud =
        nh.advertise<sensor_msgs::PointCloud2>("/multi_session/merged_map", 1, true);
    ros::Publisher pubPathRef =
        nh.advertise<nav_msgs::Path>("/multi_session/path_ref", 1, true);
    ros::Publisher pubPathCur =
        nh.advertise<nav_msgs::Path>("/multi_session/path_cur", 1, true);
    ros::Publisher pubInterLoops =
        nh.advertise<visualization_msgs::MarkerArray>("/multi_session/inter_loops", 1, true);

    // ---- 1. 加载两个会话 + 构建 session0 的 STD DB ----
    std::vector<Eigen::Affine3d> keyframe_poses_0, keyframe_poses_1;
    std::vector<PointCloud::Ptr> keyframe_clouds_0, keyframe_clouds_1;

    // 一个 STDescManager：只作为 session0 的“库”
    std::shared_ptr<STDescManager> std_manager_ref =
        std::make_shared<STDescManager>(config_setting);

    if (!loadSessionAndBuildSTD(keyframe_poses_0, keyframe_clouds_0,
                                config_setting, 0, std_manager_ref))
    {
        ROS_ERROR("Failed to load session 0");
        return -1;
    }

    // 加载 session1（这里只需要 pose + cloud，不需要 STD DB）
    if (!loadSessionAndBuildSTD(keyframe_poses_1, keyframe_clouds_1,
                                config_setting, 1, nullptr))
    {
        ROS_ERROR("Failed to load session 1");
        return -1;
    }

    // ---- 2. 真正的跨会话回环检测：session1 -> session0 ----
    std::vector<InterSessionLoop> inter_loops;
    detectInterSessionLoops(
        keyframe_clouds_1, // 当前会话：session1
        std_manager_ref,   // 库：session0
        config_setting,
        inter_loops);

    if (inter_loops.empty())
    {
        ROS_ERROR("No inter-session loops found, cannot estimate T.");
        return -1;
    }

    // ---- 3. 利用回环估计 T_W1_to_W0 （*关键使用 loop.relative_pose*）----
    Eigen::Affine3d T_w1_to_w0_prior;
    T_w1_to_w0_prior.translation() << 80, 0, 0;
    T_w1_to_w0_prior.rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));

    Eigen::Affine3d T_W1_to_W0 =
        estimateSessionTransform(inter_loops, T_w1_to_w0_prior, true, 10);

    Eigen::Matrix4d T_mat = T_W1_to_W0.matrix();
    std::cout << "Estimated T_W1_to_W0 (4x4) using STD & ICP is:\n"
              << T_mat << std::endl;
    // ---- 4. 多会话全局优化 ----
    // std::vector<Eigen::Affine3d> optimized_poses_ref, optimized_poses_cur;
    // optimizeMultiSessionPoses(
    //     keyframe_poses_0, // 已经是单会话优化的结果
    //     keyframe_poses_1, // 已经是单会话优化的结果
    //     inter_loops,
    //     T_W1_to_W0_initial, // 初始估计的变换
    //     optimized_poses_ref,
    //     optimized_poses_cur);

    // // 更新用于增量发布的位姿
    // keyframe_poses_0 = optimized_poses_ref;
    // keyframe_poses_1 = optimized_poses_cur;

    // 计算最终的会话间变换（用于点云转换）
    // Eigen::Affine3d T_W1_to_W0;
    // if (!optimized_poses_ref.empty() && !optimized_poses_cur.empty())
    // {
    //     // 使用第一个关键帧计算相对变换
    //     T_W1_to_W0 = optimized_poses_ref[0].inverse() * optimized_poses_cur[0];
    // }
    // else
    // {
    //     T_W1_to_W0 = T_W1_to_W0_initial; // 回退到初始估计
    // }

    // T_mat = T_W1_to_W0.matrix();
    // std::cout << "Optimized T_W1_to_W0 (4x4) using GTSAM is:\n"
    //           << T_mat << std::endl;

    // // 在优化后添加
    // double max_residual = 0.0;
    // for (const auto &loop : inter_loops)
    // {
    //     double residual = /* 计算回环残差 */;
    //     max_residual = std::max(max_residual, residual);
    // }
    // if (max_residual > config_setting_.max_residual_threshold_)
    // {
    //     ROS_WARN("High residual after optimization, consider re-checking loops");
    // }

    // ---- 4. 构建回环标记 ----
    visualization_msgs::MarkerArray loop_markers;
    std::string frame_id = "camera_init";
    buildInterSessionLoopMarkers(
        keyframe_poses_0,
        keyframe_poses_1,
        T_W1_to_W0,
        inter_loops,
        loop_markers,
        frame_id);

    // ---- 5. 增量发布点云和轨迹 ----
    size_t current_ref_index = 0;
    size_t current_cur_index = 0;
    ros::Rate publish_rate(10); // 10Hz发布频率
    PointCloud::Ptr merged_map(new PointCloud);
    nav_msgs::Path path_ref, path_cur;
    path_ref.header.frame_id = frame_id;
    path_cur.header.frame_id = frame_id;

    while (ros::ok())
    {
        ros::Time now = ros::Time::now();

        // 增量发布 session0 数据
        if (current_ref_index < keyframe_poses_0.size())
        {
            // 发布点云
            PointCloud transformed_cloud;
            pcl::transformPointCloud(*keyframe_clouds_0[current_ref_index],
                                     transformed_cloud,
                                     Eigen::Affine3d::Identity());
            *merged_map += transformed_cloud;

            // 发布轨迹
            geometry_msgs::PoseStamped ps;
            ps.header.frame_id = frame_id;
            ps.header.stamp = now;
            ps.pose.position.x = keyframe_poses_0[current_ref_index].translation().x();
            ps.pose.position.y = keyframe_poses_0[current_ref_index].translation().y();
            ps.pose.position.z = keyframe_poses_0[current_ref_index].translation().z();
            Eigen::Quaterniond q_ref(keyframe_poses_0[current_ref_index].rotation());
            ps.pose.orientation.w = q_ref.w();
            ps.pose.orientation.x = q_ref.x();
            ps.pose.orientation.y = q_ref.y();
            ps.pose.orientation.z = q_ref.z();
            path_ref.poses.push_back(ps);

            sensor_msgs::PointCloud2 merged_msg;
            pcl::toROSMsg(transformed_cloud, merged_msg);
            merged_msg.header.frame_id = frame_id;
            merged_msg.header.stamp = now;
            pubMergedCloud.publish(merged_msg);

            current_ref_index++;
        }

        // 增量发布 session1 数据
        if (current_cur_index < keyframe_poses_1.size())
        {
            // 发布点云
            PointCloud transformed_cloud;
            pcl::transformPointCloud(*keyframe_clouds_1[current_cur_index],
                                     transformed_cloud,
                                     T_W1_to_W0);
            *merged_map += transformed_cloud;

            // 发布轨迹
            Eigen::Affine3d T_global = T_W1_to_W0 * keyframe_poses_1[current_cur_index];
            geometry_msgs::PoseStamped ps;
            ps.header.frame_id = frame_id;
            ps.header.stamp = now;
            ps.pose.position.x = T_global.translation().x();
            ps.pose.position.y = T_global.translation().y();
            ps.pose.position.z = T_global.translation().z();
            Eigen::Quaterniond q_global(T_global.rotation());
            ps.pose.orientation.w = q_global.w();
            ps.pose.orientation.x = q_global.x();
            ps.pose.orientation.y = q_global.y();
            ps.pose.orientation.z = q_global.z();
            path_cur.poses.push_back(ps);

            sensor_msgs::PointCloud2 merged_msg;
            pcl::toROSMsg(transformed_cloud, merged_msg);
            merged_msg.header.frame_id = frame_id;
            merged_msg.header.stamp = now;
            pubMergedCloud.publish(merged_msg);

            current_cur_index++;
        }

        // 发布数据
        // sensor_msgs::PointCloud2 merged_msg;
        // pcl::toROSMsg(*merged_map, merged_msg);
        // merged_msg.header.frame_id = frame_id;
        // merged_msg.header.stamp = now;

        path_ref.header.stamp = now;
        path_cur.header.stamp = now;

        // pubMergedCloud.publish(merged_msg);
        pubPathRef.publish(path_ref);
        pubPathCur.publish(path_cur);

        // 每10帧发布一次回环标记
        if ((current_ref_index + current_cur_index) > 0 &&
            (current_ref_index + current_cur_index) % 10 == 0)
        {
            for (auto &m : loop_markers.markers)
            {
                m.header.stamp = now;
                m.header.frame_id = frame_id;
            }
            pubInterLoops.publish(loop_markers);
        }

        ros::spinOnce();
        publish_rate.sleep();
        if (current_ref_index >= keyframe_poses_0.size() && current_cur_index >= keyframe_poses_1.size())
            break;
    }
    // 保存最终地图
    pcl::io::savePCDFileASCII("/root/catkin_ws/merged_map.pcd", *merged_map);
    ROS_INFO_STREAM("Merged global map saved, size " << merged_map->size());
    return 0;
}
