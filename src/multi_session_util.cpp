#include "include/multi_session_util.h"

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
    ROS_INFO_STREAM("Start loading session" << session_id);
    keyframe_poses.clear();
    keyframe_clouds.clear();

    // 1. 决定使用哪个目录（ref / cur）
    std::string full_pose_file, full_pcd_dir, full_std_dir;
    if (session_id == 0)
    {
        full_pose_file = config_setting.ref_dir_ + "poses.txt";
        full_pcd_dir = config_setting.ref_dir_ + "pcd/";
        full_std_dir = config_setting.ref_dir_ + "std/std_database.txt";
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
    // if (session_id == 0 && std_manager_ref)
    // {
    //     std_manager_ref->loadDatabase(full_std_dir);
    //     ROS_INFO_STREAM("Database size: " << std_manager_ref->data_base_.size());
    // }
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
                      << " and loaded STD database from files.");
        ROS_INFO_STREAM("STD database size " << std_manager_ref->data_base_.size());
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
            std_manager_ref->data_base_, 1);

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

bool detectInterSessionLoopForKeyframe(
    int cur_kf,
    const std::vector<STDesc> &cur_stds,
    const std::shared_ptr<STDescManager> &std_manager_ref,
    const ConfigSetting &config_setting,
    InterSessionLoop &out_loop)
{
    out_loop = InterSessionLoop(); // 清空输出

    if (!std_manager_ref)
    {
        ROS_ERROR("detectInterSessionLoopForKeyframe: std_manager_ref is null");
        return false;
    }

    if (cur_stds.empty())
    {
        ROS_WARN_STREAM("detectInterSessionLoopForKeyframe: cur_stds empty at kf "
                        << cur_kf << ", skip.");
        return false;
    }

    // 当前 query 对应的平面索引：GenerateSTDescs 之后 plane_cloud_vec_ 的最后一个
    int query_plane_index =
        static_cast<int>(std_manager_ref->plane_cloud_vec_.size()) - 1;
    if (query_plane_index < 0)
    {
        ROS_ERROR("detectInterSessionLoopForKeyframe: plane_cloud_vec_ is empty; "
                  "make sure you called GenerateSTDescs before this function.");
        return false;
    }

    // 为了绕过 skip_near_num_，把 query 的 frame_id_ 设成一个很大的值
    const int LARGE_FRAME_ID_BASE = 1000000;
    int query_frame_id_for_skip = LARGE_FRAME_ID_BASE + cur_kf;

    // 复制一份 STDesc，并修改 frame_id_
    std::vector<STDesc> stds_vec = cur_stds;
    for (auto &d : stds_vec)
        d.frame_id_ = query_frame_id_for_skip;

    // 当前帧在 plane_cloud_vec_ 里的索引（目前 SearchLoop 里主要还是用 back()）
    std_manager_ref->current_frame_id_ = query_plane_index;

    // 在 session0 的 STD 数据库上做回环搜索
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
        out_loop.ref_kf = ref_kf;
        out_loop.cur_kf = cur_kf;
        out_loop.score = score;
        out_loop.relative_pose = loop_transform; // T_cur_to_ref
        out_loop.match_pairs = loop_std_pair;

        ROS_INFO_STREAM("[Inter-Loop-Online] cur_kf " << cur_kf
                                                      << " -> ref_kf " << ref_kf
                                                      << ", score = " << score
                                                      << ", matches = " << loop_std_pair.size());
        return true;
    }

    // 没通过阈值，当作未检测到 inter-session loop
    return false;
}

// 利用 inter_loops 里每条 loop 的 relative_pose（STD + plane ICP 得到的）
// 直接估计 T_W1_to_W0
Eigen::Affine3d estimateSessionTransform(
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

void visualizeMultiSessionLoop(
    const ros::Publisher &publisher,
    const Eigen::Affine3d &pose_cur,
    const Eigen::Affine3d &pose_ref)
{
    visualization_msgs::MarkerArray markerArray;

    // 节点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "camera_init";
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "ms_loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1.0;
    markerNode.scale.x = 0.4;
    markerNode.scale.y = 0.4;
    markerNode.scale.z = 0.4;
    markerNode.color.r = 1.0;
    markerNode.color.g = 0.0;
    markerNode.color.b = 0.0;
    markerNode.color.a = 1.0;

    // 边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "camera_init";
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "ms_loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1.0;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 1.0;
    markerEdge.color.g = 0.0;
    markerEdge.color.b = 0.0;
    markerEdge.color.a = 1.0;

    geometry_msgs::Point p_cur, p_ref;
    p_cur.x = pose_cur.translation().x();
    p_cur.y = pose_cur.translation().y();
    p_cur.z = pose_cur.translation().z();

    p_ref.x = pose_ref.translation().x();
    p_ref.y = pose_ref.translation().y();
    p_ref.z = pose_ref.translation().z();

    markerNode.points.push_back(p_cur);
    markerNode.points.push_back(p_ref);
    markerEdge.points.push_back(p_cur);
    markerEdge.points.push_back(p_ref);

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    publisher.publish(markerArray);
}
