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
using PointType = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointType>;

// ----------------------------------
// 功能1：加载某个会话的关键帧 pose + 点云
//       若 session_id==0，则同时用点云构建 STD 数据库
// ----------------------------------
bool loadSessionAndBuildSTD(
    std::vector<Eigen::Affine3d> &keyframe_poses,
    std::vector<PointCloud::Ptr> &keyframe_clouds,
    const ConfigSetting &config_setting,
    int session_id,
    const std::shared_ptr<STDescManager> &std_manager_ref)
{
    keyframe_poses.clear();
    keyframe_clouds.clear();

    // 1. 决定使用哪个目录（ref / cur）
    std::string full_pose_file, full_pcd_dir;
    if (session_id == 0)
    {
        full_pose_file = config_setting.ref_dir_ + "poses.txt";
        full_pcd_dir = config_setting.ref_dir_ + "pcd/";
    }
    else
    {
        full_pose_file = config_setting.cur_dir_ + "poses.txt";
        full_pcd_dir = config_setting.cur_dir_ + "pcd/";
    }

    std::ifstream fin(full_pose_file);
    if (!fin.is_open())
    {
        ROS_ERROR_STREAM("Cannot open pose file: " << full_pose_file);
        return false;
    }

    std::string line;
    int idx = 0;

    while (std::getline(fin, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        double x, y, z;
        double qw, qx, qy, qz;
        char comma;

        // 解析格式：x, y, z, qw, qx, qy, qz
        ss >> x >> comma >> y >> comma >> z >> comma >> qw >> comma >> qx >> comma >> qy >> comma >> qz;

        Eigen::Quaterniond q(qw, qx, qy, qz);
        q.normalize();

        Eigen::Affine3d T = Eigen::Affine3d::Identity();
        T.translation() = Eigen::Vector3d(x, y, z);
        T.linear() = q.toRotationMatrix();
        keyframe_poses.push_back(T);

        // 对应 PCD 文件：0000.pcd, 0001.pcd, ...
        std::ostringstream oss;
        oss << std::setw(4) << std::setfill('0') << idx;
        std::string pcd_path = full_pcd_dir + oss.str() + ".pcd";

        PointCloud::Ptr cloud(new PointCloud);
        if (pcl::io::loadPCDFile(pcd_path, *cloud) != 0)
        {
            ROS_ERROR_STREAM("Failed to load PCD file: " << pcd_path);
            return false;
        }
        keyframe_clouds.push_back(cloud);

        // 只有 session0 需要把 STD 放进数据库，作为“库”
        if (session_id == 0 && std_manager_ref)
        {
            std::vector<STDesc> stds_vec;
            // GenerateSTDescs 会：
            //   1) 提取平面并 push 到 plane_cloud_vec_
            //   2) 提取 corner
            //   3) 生成 ST 描述子
            std_manager_ref->GenerateSTDescs(cloud, stds_vec);

            // 离线更直观：强制让 frame_id_ = 关键帧索引
            for (auto &d : stds_vec)
                d.frame_id_ = idx;

            std_manager_ref->AddSTDescs(stds_vec);
        }

        ++idx;
    }

    fin.close();

    if (session_id == 0)
    {
        ROS_INFO_STREAM(
            "Loaded " << keyframe_poses.size()
                      << " keyframes from " << full_pose_file
                      << " and built STD database from point clouds.");
    }
    else
    {
        ROS_INFO_STREAM(
            "Loaded " << keyframe_poses.size()
                      << " keyframes from " << full_pose_file
                      << " (poses + clouds only, no STD DB).");
    }

    return true;
}

// ----------------------------------
// 功能2：跨会话回环检测
//
// 设计要点：
//   * 只在 std_manager_ref 中保存 session0 的 DB + 所有平面
//   * session1 仅用点云产生 query 描述子，不 AddSTDescs
//   * 对于每个 query keyframe：
//       - GenerateSTDescs -> 追加一个平面到 plane_cloud_vec_ 的末尾
//       - 把 query STD 的 frame_id_ 设成很大，以绕开 skip_near_num_
//       - 调用 SearchLoop(stds_vec, ..., std_manager_ref->data_base_)
//   * SearchLoop 内部：
//       - candidate_selector 在 DB（session0）里收集候选帧
//       - candidate_verify 调用 plane_geometric_verify(
//             plane_cloud_vec_.back() 作为 source(=query),
//             plane_cloud_vec_[candidate_id] 作为 target(=库的关键帧平面)
//         得到 verify_score 和 loop_transform
// ----------------------------------
bool detectInterSessionLoops(
    const std::vector<PointCloud::Ptr> &keyframe_clouds_cur,
    const std::shared_ptr<STDescManager> &std_manager_ref,
    const ConfigSetting &config_setting,
    std::vector<InterSessionLoop> &inter_loops)
{
    inter_loops.clear();

    if (!std_manager_ref)
    {
        ROS_ERROR("detectInterSessionLoops: std_manager_ref is null");
        return false;
    }

    const int num_kf = static_cast<int>(keyframe_clouds_cur.size());
    if (num_kf == 0)
    {
        ROS_WARN("detectInterSessionLoops: no keyframes in current session.");
        return false;
    }

    ROS_INFO_STREAM("detectInterSessionLoops: #cur keyframes = " << num_kf
                                                                 << ", ref DB frames (plane_cloud_vec_) = "
                                                                 << std_manager_ref->plane_cloud_vec_.size());

    const int LARGE_FRAME_ID_BASE = 1000000; // 绕过 skip_near_num_
    const int step = 1;                      // 可改成 >1 做 subsampling

    for (int k1 = 0; k1 < num_kf; k1 += step)
    {
        if (!keyframe_clouds_cur[k1])
        {
            ROS_WARN_STREAM("detectInterSessionLoops: keyframe cloud " << k1
                                                                       << " is null, skip.");
            continue;
        }

        // A. 复制一份点云（防止 const 问题）
        PointCloud::Ptr cloud_k1(new PointCloud);
        *cloud_k1 = *keyframe_clouds_cur[k1];

        // B. 用 session1 的 keyframe 点云生成 STD (query)
        std::vector<STDesc> stds_vec;
        std_manager_ref->GenerateSTDescs(cloud_k1, stds_vec);

        if (stds_vec.empty())
        {
            ROS_WARN_STREAM("Keyframe " << k1 << " has no STDesc, skip.");
            continue;
        }

        // 当前 query 对应的平面索引，是 plane_cloud_vec_ 的最后一个
        int query_plane_index =
            static_cast<int>(std_manager_ref->plane_cloud_vec_.size()) - 1;
        if (query_plane_index < 0)
        {
            ROS_ERROR("detectInterSessionLoops: plane_cloud_vec_ is empty after GenerateSTDescs.");
            continue;
        }

        // 为了绕过 skip_near_num_，把 query 的 frame_id_ 设成一个很大的值
        int query_frame_id_for_skip = LARGE_FRAME_ID_BASE + k1;
        for (auto &d : stds_vec)
            d.frame_id_ = query_frame_id_for_skip;

        // （目前 candidate_verify 使用 plane_cloud_vec_.back()
        //  作为 source，不使用 current_frame_id_，因此这里设置与否影响不大）
        std_manager_ref->current_frame_id_ = query_plane_index;

        // C. 在 session0 的 STD 数据库上做回环搜索
        std::pair<int, double> loop_result(-1, 0.0);
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        loop_transform.first.setZero();
        loop_transform.second.setIdentity();
        std::vector<std::pair<STDesc, STDesc>> loop_std_pair;

        std_manager_ref->SearchLoop(
            stds_vec,
            loop_result,
            loop_transform,
            loop_std_pair,
            std_manager_ref->data_base_);

        int ref_kf = loop_result.first;    // session0 的关键帧索引
        double score = loop_result.second; // 平面 ICP 得到的匹配得分

        if (ref_kf >= 0 && score > config_setting.icp_threshold_)
        {
            InterSessionLoop loop;
            loop.ref_kf = ref_kf;
            loop.cur_kf = k1;
            loop.score = score;
            loop.relative_pose = loop_transform; // 关键：T_F0_F1（后面要用）
            loop.match_pairs = loop_std_pair;

            inter_loops.push_back(loop);

            ROS_INFO_STREAM("[Inter-Loop] cur_kf " << k1
                                                   << " -> ref_kf " << ref_kf
                                                   << ", score = " << score
                                                   << ", matches = " << loop_std_pair.size());
            // k1 += 5; // 回环成功后跳几帧
        }
    }

    ROS_INFO_STREAM("detectInterSessionLoops: found "
                    << inter_loops.size() << " inter-session loops.");

    return !inter_loops.empty();
}

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

// 利用 inter_loops 里每条 loop 的 relative_pose（STD + plane ICP 得到的）
// 直接估计 T_W1_to_W0
Eigen::Affine3d estimateSessionTransformFromLoops_STD(
    const std::vector<InterSessionLoop> &inter_loops,
    bool yaw_only = true) // 默认先用 yaw + 平移，环境近似平面
{
    Eigen::Affine3d T_W1_to_W0 = Eigen::Affine3d::Identity();

    if (inter_loops.empty())
    {
        std::cerr << "[estimateSessionTransformFromLoops_STD] No loops, return Identity.\n";
        return T_W1_to_W0;
    }

    // ------- 收集所有 loop 的 R, t, w -------
    struct LoopRT
    {
        double yaw;
        Eigen::Vector3d t;
        double w;
        Eigen::Matrix3d R;
    };

    std::vector<LoopRT> data;
    data.reserve(inter_loops.size());

    for (const auto &loop : inter_loops)
    {
        LoopRT item;
        item.R = loop.relative_pose.second;
        item.t = loop.relative_pose.first;
        item.w = std::max(loop.score, 1e-3); // 用 icp score 作为权重

        // 从 R 里提 yaw（ZYX: R = Rz(yaw)*Ry(pitch)*Rx(roll)）
        const Eigen::Matrix3d &R = item.R;
        double yaw = std::atan2(R(1, 0), R(0, 0));
        item.yaw = yaw;

        data.push_back(item);
    }

    // (可选) 简单的 outlier 剔除：按 yaw 的中值做一个粗过滤
    {
        std::vector<double> yaw_list;
        yaw_list.reserve(data.size());
        for (auto &d : data)
            yaw_list.push_back(d.yaw);

        std::sort(yaw_list.begin(), yaw_list.end());
        double yaw_med = yaw_list[yaw_list.size() / 2];

        const double yaw_thresh = 20.0 * M_PI / 180.0; // 例如 20 度以内保留
        std::vector<LoopRT> filtered;
        for (auto &d : data)
        {
            double diff = std::fabs(d.yaw - yaw_med);
            // 处理一下 2π 周期
            if (diff > M_PI)
                diff = 2 * M_PI - diff;

            if (diff < yaw_thresh)
                filtered.push_back(d);
        }

        if (!filtered.empty())
            data.swap(filtered);
    }

    if (data.empty())
    {
        std::cerr << "[estimateSessionTransformFromLoops_STD] All loops filtered out, return Identity.\n";
        return T_W1_to_W0;
    }

    // ------- 估计旋转 -------
    Eigen::Matrix3d R_avg = Eigen::Matrix3d::Identity();

    if (!yaw_only)
    {
        // 完整 3D 旋转加权平均（四元数）
        Eigen::Quaterniond q_ref(data.front().R);
        Eigen::Vector4d q_sum(0, 0, 0, 0);
        double w_sum = 0.0;

        for (auto &d : data)
        {
            Eigen::Quaterniond q_i(d.R);
            double w = d.w;
            if (q_ref.dot(q_i) < 0.0)
                q_i.coeffs() *= -1.0; // 对齐符号，防止 q / -q 抵消

            q_sum += w * q_i.coeffs();
            w_sum += w;
        }

        q_sum /= std::max(w_sum, 1e-9);
        Eigen::Quaterniond q_avg(q_sum);
        q_avg.normalize();
        R_avg = q_avg.toRotationMatrix();
    }
    else
    {
        // 只平均 yaw（绕 z），避免 pitch/roll 漂
        double cos_sum = 0.0, sin_sum = 0.0;
        double w_sum = 0.0;

        for (auto &d : data)
        {
            double w = d.w;
            cos_sum += w * std::cos(d.yaw);
            sin_sum += w * std::sin(d.yaw);
            w_sum += w;
        }

        double yaw_avg = std::atan2(sin_sum, cos_sum);

        double cy = std::cos(yaw_avg);
        double sy = std::sin(yaw_avg);
        R_avg << cy, -sy, 0,
            sy, cy, 0,
            0, 0, 1;

        std::cout << "[estimateSessionTransformFromLoops_STD] use "
                  << data.size() << " loops, yaw-only:\n"
                  << "  yaw (deg) = " << yaw_avg * 180.0 / M_PI << std::endl;
    }

    // ------- 估计平移 -------
    Eigen::Vector3d t_sum(0, 0, 0);
    double t_w_sum = 0.0;
    for (auto &d : data)
    {
        t_sum += d.w * d.t;
        t_w_sum += d.w;
    }
    Eigen::Vector3d t_avg = t_sum / std::max(t_w_sum, 1e-9);

    T_W1_to_W0 = Eigen::Affine3d::Identity();
    T_W1_to_W0.linear() = R_avg;
    T_W1_to_W0.translation() = t_avg;

    std::cout << "[estimateSessionTransformFromLoops_STD] final T_W1_to_W0 (from STD relative poses):\n"
              << T_W1_to_W0.matrix() << std::endl;

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

    // 做一次 self-loop 检查（session0 vs session0），看 pipeline 是否 OK
    // std::vector<InterSessionLoop> self_loops;
    // detectInterSessionLoops(
    //     keyframe_clouds_0, // 把 session0 当成“当前会话”
    //     std_manager_ref,   // 也是“参考库”
    //     config_setting,
    //     self_loops);
    // ROS_INFO_STREAM("[Self-loop check] session0 vs session0, loops = "
    //                 << self_loops.size());

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
    // Eigen::Affine3d T_W1_to_W0 =
    //     estimateSessionTransformFromLoops(keyframe_poses_0,
    //                                       keyframe_poses_1,
    //                                       inter_loops);
    Eigen::Affine3d T_W1_to_W0 =
        estimateSessionTransformFromLoops_STD(inter_loops, true);

    Eigen::Matrix4d T_mat = T_W1_to_W0.matrix();
    std::cout << "Estimated T_W1_to_W0 (4x4):\n"
              << T_mat << std::endl;

    // ---- 4. 构建合并点云和轨迹 + loop 的 Marker ----
    PointCloud::Ptr merged_map;
    nav_msgs::Path path_ref, path_cur;
    std::string frame_id = "camera_init";

    buildMergedMapAndPaths(
        keyframe_poses_0, keyframe_clouds_0,
        keyframe_poses_1, keyframe_clouds_1,
        T_W1_to_W0,
        merged_map,
        path_ref,
        path_cur,
        frame_id);
    pcl::io::savePCDFileASCII ("/root/catkin_ws/merged_map.pcd", *merged_map);

    visualization_msgs::MarkerArray loop_markers;
    buildInterSessionLoopMarkers(
        keyframe_poses_0,
        keyframe_poses_1,
        T_W1_to_W0,
        inter_loops,
        loop_markers,
        frame_id);

    // ---- 5. 转成 ROS 消息并循环发布 ----
    sensor_msgs::PointCloud2 merged_msg;
    pcl::toROSMsg(*merged_map, merged_msg);
    merged_msg.header.frame_id = frame_id;

    ros::Rate rate(1.0); // 1Hz 刷新
    while (ros::ok())
    {
        ros::Time now = ros::Time::now();

        merged_msg.header.stamp = now;
        path_ref.header.frame_id = frame_id;
        path_cur.header.frame_id = frame_id;
        path_ref.header.stamp = now;
        path_cur.header.stamp = now;

        for (auto &m : loop_markers.markers)
        {
            m.header.stamp = now;
            m.header.frame_id = frame_id;
        }

        pubMergedCloud.publish(merged_msg);
        pubPathRef.publish(path_ref);
        pubPathCur.publish(path_cur);
        pubInterLoops.publish(loop_markers);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
