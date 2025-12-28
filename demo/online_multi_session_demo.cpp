/**
 * Online Multi-Session Demo (v1.2 Revised)
 * Base: v1.1 (Tightly Coupled Graph)
 * Changes:
 * 1. Load initial T from YAML (vector<double>).
 * 2. Loose prior on S1 start (Bolder update).
 * 3. Safe Session 0 map publishing (Merged & Downsampled).
 */

#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include "../include/STDesc.h"
#include "../include/multi_session_util.h"
#include <fstream>

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> PointCloud;

std::mutex laser_mtx;
std::mutex odom_mtx;

std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;

void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  std::unique_lock<std::mutex> lock(laser_mtx);
  laser_buffer.push(msg);
}

void OdomHandler(const nav_msgs::Odometry::ConstPtr &msg)
{
  std::unique_lock<std::mutex> lock(odom_mtx);
  odom_buffer.push(msg);
}

bool syncPackages(PointCloud::Ptr &cloud, Eigen::Affine3d &pose)
{
  if (laser_buffer.empty() || odom_buffer.empty())
    return false;

  auto laser_msg = laser_buffer.front();
  double laser_timestamp = laser_msg->header.stamp.toSec();

  auto odom_msg = odom_buffer.front();
  double odom_timestamp = odom_msg->header.stamp.toSec();

  if (abs(odom_timestamp - laser_timestamp) < 1e-3)
  {
    pcl::fromROSMsg(*laser_msg, *cloud);

    Eigen::Quaterniond r(
        odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
    Eigen::Vector3d t(odom_msg->pose.pose.position.x,
                      odom_msg->pose.pose.position.y,
                      odom_msg->pose.pose.position.z);

    pose = Eigen::Affine3d::Identity();
    pose.translate(t);
    pose.rotate(r);

    std::unique_lock<std::mutex> l_lock(laser_mtx);
    std::unique_lock<std::mutex> o_lock(odom_mtx);

    laser_buffer.pop();
    odom_buffer.pop();
  }
  else if (odom_timestamp < laser_timestamp)
  {
    std::unique_lock<std::mutex> o_lock(odom_mtx);
    odom_buffer.pop();
    return false;
  }
  else
  {
    std::unique_lock<std::mutex> l_lock(laser_mtx);
    laser_buffer.pop();
    return false;
  }

  return true;
}

void visualizeLoopClosure(
    const ros::Publisher &publisher,
    const std::vector<std::pair<int, int>> &loop_container,
    const std::vector<Eigen::Affine3d> &key_pose_vec)
{
  if (loop_container.empty())
    return;

  visualization_msgs::MarkerArray markerArray;
  visualization_msgs::Marker markerNode;
  markerNode.header.frame_id = "session0_init";
  markerNode.action = visualization_msgs::Marker::ADD;
  markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
  markerNode.ns = "loop_nodes";
  markerNode.id = 0;
  markerNode.pose.orientation.w = 1;
  markerNode.scale.x = 0.3;
  markerNode.scale.y = 0.3;
  markerNode.scale.z = 0.3;
  markerNode.color.r = 0;
  markerNode.color.g = 0.8;
  markerNode.color.b = 1;
  markerNode.color.a = 1;

  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "session0_init";
  markerEdge.action = visualization_msgs::Marker::ADD;
  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge.ns = "loop_edges";
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1;
  markerEdge.scale.x = 0.1;
  markerEdge.color.r = 0.9;
  markerEdge.color.g = 0.9;
  markerEdge.color.b = 0;
  markerEdge.color.a = 1;

  for (auto it = loop_container.begin(); it != loop_container.end(); ++it)
  {
    int key_cur = it->first;
    int key_pre = it->second;

    // Bounds check
    if (key_cur >= key_pose_vec.size() || key_pre >= key_pose_vec.size())
      continue;

    geometry_msgs::Point p;
    p.x = key_pose_vec[key_cur].translation().x();
    p.y = key_pose_vec[key_cur].translation().y();
    p.z = key_pose_vec[key_cur].translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
    p.x = key_pose_vec[key_pre].translation().x();
    p.y = key_pose_vec[key_pre].translation().y();
    p.z = key_pose_vec[key_pre].translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
  }

  markerArray.markers.push_back(markerNode);
  markerArray.markers.push_back(markerEdge);
  publisher.publish(markerArray);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "online_multi_session_demo");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  ConfigSetting config_setting;
  read_parameters(nh, config_setting);
  static const std::string W0_FRAME = "session0_init";
  static const std::string W1_FRAME = "camera_init";

  ros::Publisher pubCurrentCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  ros::Publisher pubCurrentCorner = nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
  ros::Publisher pubMatchedCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
  ros::Publisher pubMatchedCorner = nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
  ros::Publisher pubSTD = nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

  ros::Publisher pubOriginCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_origin", 10000);
  ros::Publisher pubCorrectCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 10000);
  ros::Publisher pubCorrectPath = nh.advertise<nav_msgs::Path>("/correct_path", 100000);
  ros::Publisher pubOdomOrigin = nh.advertise<nav_msgs::Odometry>("/odom_origin", 10);
  ros::Publisher pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 10);
  ros::Publisher pubSession0Cloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_session0", 1, true);
  ros::Publisher pubSession0Path = nh.advertise<nav_msgs::Path>("/path_session0", 1, true);

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered_body", 100, laserCloudHandler);
  ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, OdomHandler);

  STDescManager *std_manager = new STDescManager(config_setting);
  std::shared_ptr<STDescManager> std_manager_prior(new STDescManager(config_setting));

  gtsam::Values initial;
  gtsam::NonlinearFactorGraph graph;

  // ==========================================
  // 1. 读取 YAML 先验矩阵 & 默认值设置
  // ==========================================
  Eigen::Affine3d T_W1_to_W0_est = Eigen::Affine3d::Identity();
  std::vector<double> matrix_vals;
  // 从参数服务器读取 16 个 double (行优先)
  if (nh.getParam("multi_session/initial_T_W1_W0", matrix_vals) && matrix_vals.size() == 16)
  {
    ROS_INFO("Loaded initial T_W1_W0 from ROS params.");
    Eigen::Matrix4d mat;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        mat(i, j) = matrix_vals[i * 4 + j];
    T_W1_to_W0_est = Eigen::Affine3d(mat);
  }
  else
  {
    ROS_WARN("Failed to load multi_session/initial_T_W1_W0, using default 80m offset.");
    T_W1_to_W0_est.translation() << 80, 0, 0;
    T_W1_to_W0_est.rotate(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
  }

  std::cout << "Initial Guess T (W1->W0):\n"
            << T_W1_to_W0_est.matrix() << std::endl;

  // ==========================================
  // 2. 配置噪声模型 (Bold Update 策略)
  // ==========================================
  // 里程计噪声
  auto odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
  // 固定锚点噪声 (Session 0)
  auto priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished());

  auto s1StartPriorNoise = gtsam::noiseModel::Diagonal::Variances(
      (gtsam::Vector(6) << 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0).finished());

  double intraScore = 0.01;
  gtsam::Vector robustNoiseVector6(6);
  robustNoiseVector6 << intraScore, intraScore, intraScore, intraScore, intraScore, intraScore;
  auto robustLoopNoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(1),
      gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

  double interScore = 0.001;
  auto interLoopNoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(1),
      gtsam::noiseModel::Diagonal::Variances(gtsam::Vector6::Constant(interScore)));

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  gtsam::ISAM2 isam(parameters);

  size_t cloudInd = 0;
  size_t keyCloudInd = 0;

  std::vector<PointCloud::Ptr> cloud_vec;
  std::vector<Eigen::Affine3d> pose_vec;
  std::vector<Eigen::Affine3d> origin_pose_vec;
  std::vector<std::pair<int, int>> loop_container;
  // [修复] 补上关键变量定义
  std::vector<Eigen::Affine3d> keyframe_pose_vec;

  std::vector<Eigen::Affine3d> ref_keyframe_poses;
  std::vector<PointCloud::Ptr> ref_keyframe_clouds;

  PointCloud::Ptr key_cloud(new PointCloud);
  bool has_loop_flag = false;
  gtsam::Values curr_estimate;

  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::WallRate rate(100.0);

  bool keyframes_saved = false;
  ros::WallTime last_data_time = ros::WallTime::now();

  std::ofstream loop_log_file;
  nav_msgs::Path path_s0; // S0 Path 缓存

  // ==========================================
  // 3. 加载 Session 0 (带修复的发布逻辑)
  // ==========================================
  if (config_setting.multi_session_mode_ == 1)
  {
    if (!loadSessionAndBuildSTD(ref_keyframe_poses, ref_keyframe_clouds, config_setting, 0, std_manager_prior))
    {
      ROS_ERROR("Failed to load reference session.");
    }
    else
    {
      ROS_INFO("Reference session loaded, keyframes = %zu", ref_keyframe_poses.size());

      // [修复] 合并 Session 0 点云并一次性发布，避免循环 sleep 卡死或 OOM
      PointCloud::Ptr full_s0(new PointCloud);
      for (auto &c : ref_keyframe_clouds)
      {
        if (c && !c->empty())
          *full_s0 += *c;
      }
      if (!full_s0->empty())
      {
        down_sampling_voxel(*full_s0, 0.2); // 降采样以用于可视化
        sensor_msgs::PointCloud2 map_msg;
        pcl::toROSMsg(*full_s0, map_msg);
        map_msg.header.frame_id = W0_FRAME;
        map_msg.header.stamp = ros::Time::now();
        pubSession0Cloud.publish(map_msg);
        ROS_INFO("Published merged Session 0 map (size: %zu)", full_s0->size());
      }

      // 添加 Session 0 到因子图 (Fixed Anchors)
      for (size_t i = 0; i < ref_keyframe_poses.size(); ++i)
      {
        initial.insert(gtsam::Symbol('a', i), gtsam::Pose3(ref_keyframe_poses[i].matrix()));
      }
      graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('a', 0), gtsam::Pose3(ref_keyframe_poses[0].matrix()), priorNoise));
      isam.update(graph, initial);
      graph.resize(0);
      initial.clear();

      // 构建 Session 0 路径
      path_s0.header.frame_id = W0_FRAME;
      path_s0.header.stamp = ros::Time::now();
      for (const auto &pose : ref_keyframe_poses)
      {
        geometry_msgs::PoseStamped ps;
        ps.header.frame_id = W0_FRAME;
        ps.pose.position.x = pose.translation().x();
        ps.pose.position.y = pose.translation().y();
        ps.pose.position.z = pose.translation().z();
        Eigen::Quaterniond q(pose.rotation());
        ps.pose.orientation.w = q.w();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        path_s0.poses.push_back(ps);
      }
      pubSession0Path.publish(path_s0);
      ROS_INFO("Session 0 path published.");
    }
  }

  // 初始化日志
  loop_log_file.open(config_setting.cur_dir_ + "inter_session_loops.log");
  if (loop_log_file.is_open())
    loop_log_file << "# TS, S1_KF, S0_KF, Score\n";

  tf2_ros::TransformBroadcaster tf_br;
  auto publishW0W1TF = [&](const Eigen::Affine3d &T_est)
  {
    if (!T_est.matrix().allFinite())
      return;
    Eigen::Affine3d T_W0_to_W1 = T_est.inverse();
    Eigen::Quaterniond q(T_W0_to_W1.rotation());
    Eigen::Vector3d t = T_W0_to_W1.translation();
    if (!t.allFinite() || !q.coeffs().allFinite())
      return;
    q.normalize();
    geometry_msgs::TransformStamped tf_msg;
    tf_msg.header.stamp = ros::Time::now();
    tf_msg.header.frame_id = W0_FRAME;
    tf_msg.child_frame_id = W1_FRAME;
    tf_msg.transform.translation.x = t.x();
    tf_msg.transform.translation.y = t.y();
    tf_msg.transform.translation.z = t.z();
    tf_msg.transform.rotation.w = q.w();
    tf_msg.transform.rotation.x = q.x();
    tf_msg.transform.rotation.y = q.y();
    tf_msg.transform.rotation.z = q.z();
    tf_br.sendTransform(tf_msg);
  };

  while (ros::ok())
  {
    PointCloud::Ptr current_cloud_body(new PointCloud);
    PointCloud::Ptr current_cloud_world(new PointCloud);
    Eigen::Affine3d pose_local;

    if (syncPackages(current_cloud_body, pose_local))
    {
      publishW0W1TF(T_W1_to_W0_est);
      last_data_time = ros::WallTime::now();

      // 使用读取/优化后的 T 计算全局初值
      auto pose_global_guess = T_W1_to_W0_est * pose_local;

      pcl::transformPointCloud(*current_cloud_body, *current_cloud_world, pose_local);
      down_sampling_voxel(*current_cloud_world, config_setting.ds_size_);
      cloud_vec.push_back(current_cloud_body);

      pose_vec.push_back(pose_global_guess); // 存入 W0 下的 Pose
      origin_pose_vec.push_back(pose_local); // 存入 W1 下的原始 Odom

      sensor_msgs::PointCloud2 pub_cloud;
      pcl::toROSMsg(*current_cloud_world, pub_cloud);
      pub_cloud.header.frame_id = W1_FRAME; // Raw data in W1
      pub_cloud.header.stamp = ros::Time::now();
      pubOriginCloud.publish(pub_cloud);

      Eigen::Quaterniond _r(pose_local.rotation());
      nav_msgs::Odometry odom;
      odom.header.frame_id = W1_FRAME;
      odom.header.stamp = ros::Time::now();
      odom.pose.pose.position.x = pose_local.translation().x();
      odom.pose.pose.position.y = pose_local.translation().y();
      odom.pose.pose.position.z = pose_local.translation().z();
      odom.pose.pose.orientation.w = _r.w();
      odom.pose.pose.orientation.x = _r.x();
      odom.pose.pose.orientation.y = _r.y();
      odom.pose.pose.orientation.z = _r.z();
      pubOdomOrigin.publish(odom);

      *key_cloud += *current_cloud_world;

      // 插入因子图 (W0 frame)
      initial.insert(gtsam::Symbol('b', cloudInd), gtsam::Pose3(pose_global_guess.matrix()));

      if (!cloudInd)
      {
        // [关键] 使用松弛的 s1StartPriorNoise
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(
            gtsam::Symbol('b', 0), gtsam::Pose3(pose_global_guess.matrix()), s1StartPriorNoise));
      }
      else
      {
        auto prev_pose = gtsam::Pose3(origin_pose_vec[cloudInd - 1].matrix());
        auto curr_pose = gtsam::Pose3(pose_local.matrix());
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            gtsam::Symbol('b', cloudInd - 1), gtsam::Symbol('b', cloudInd), prev_pose.between(curr_pose),
            odometryNoise));
      }

      if (cloudInd % config_setting.sub_frame_num_ == 0 && cloudInd != 0)
      {
        ROS_INFO("key frame idx: [%d]", (int)keyCloudInd);
        std::vector<STDesc> stds_vec;
        std_manager->GenerateSTDescs(key_cloud, stds_vec);

        // Intra-session
        std::pair<int, double> search_result(-1, 0);
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        loop_transform.first << 0, 0, 0;
        loop_transform.second = Eigen::Matrix3d::Identity();
        std::vector<std::pair<STDesc, STDesc>> loop_std_pair;

        if (keyCloudInd > config_setting.skip_near_num_)
        {
          std_manager->SearchLoop(stds_vec, search_result, loop_transform, loop_std_pair, std_manager->data_base_);
        }
        std_manager->AddSTDescs(stds_vec);

        // Inter-session
        std::pair<int, double> ms_search_result(-1, 0);
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> ms_loop_transform;
        std::vector<std::pair<STDesc, STDesc>> ms_loop_std_pair;
        if (config_setting.multi_session_mode_ == 1)
        {
          std::vector<STDesc> ms_stds_vec;
          // key_cloud is local (W1). Descriptors are rotation invariant locally.
          std_manager_prior->GenerateSTDescs(key_cloud, ms_stds_vec);

          if (!ms_stds_vec.empty())
          {
            std_manager_prior->SearchLoop(ms_stds_vec, ms_search_result, ms_loop_transform, ms_loop_std_pair, std_manager_prior->data_base_, 1);

            int ms_match_kf = ms_search_result.first;
            double ms_score = ms_search_result.second;

            if (ms_match_kf >= 0 && ms_score > config_setting.inter_session_icp_threshold_)
            {
              has_loop_flag = true;

              // ICP Refine (S1 -> S0)
              std_manager_prior->PlaneGeometricIcp(
                  std_manager_prior->plane_cloud_vec_.back(),
                  std_manager_prior->plane_cloud_vec_[ms_match_kf],
                  ms_loop_transform);

              gtsam::Pose3 relative_pose(gtsam::Rot3(ms_loop_transform.second), gtsam::Point3(ms_loop_transform.first));

              graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                  gtsam::Symbol('a', ms_match_kf), gtsam::Symbol('b', keyCloudInd),
                  relative_pose, interLoopNoise));

              ROS_INFO_STREAM("[MS Loop Factor]: S1_" << keyCloudInd << " -> S0_" << ms_match_kf << " score " << ms_score);
              if (loop_log_file.is_open())
                loop_log_file << ros::Time::now() << ", " << keyCloudInd << ", " << ms_match_kf << ", " << ms_score << std::endl;
            }
          }
        }

        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(*key_cloud, pub_cloud);
        pub_cloud.header.frame_id = W1_FRAME;
        pub_cloud.header.stamp = ros::Time::now();
        pubCurrentCloud.publish(pub_cloud);

        std_manager->key_cloud_vec_.push_back(key_cloud->makeShared());
        keyframe_pose_vec.push_back(pose_global_guess);

        if (search_result.first > 0)
        {
          has_loop_flag = true;
          int match_frame = search_result.first;
          std_manager->PlaneGeometricIcp(std_manager->plane_cloud_vec_.back(), std_manager->plane_cloud_vec_[match_frame], loop_transform);
          gtsam::Pose3 relative_pose(gtsam::Rot3(loop_transform.second), gtsam::Point3(loop_transform.first));
          graph.add(gtsam::BetweenFactor<gtsam::Pose3>(gtsam::Symbol('b', match_frame), gtsam::Symbol('b', keyCloudInd), relative_pose, robustLoopNoise));
          publish_std_pairs(loop_std_pair, pubSTD);
        }

        key_cloud->clear();
        ++keyCloudInd;
      }

      // Optimize
      isam.update(graph, initial);
      isam.update();
      if (has_loop_flag)
      {
        for (int k = 0; k < 5; ++k)
          isam.update();
      }
      graph.resize(0);
      initial.clear();

      curr_estimate = isam.calculateEstimate();

      // Update Pose Vec (Now in W0 frame)
      for (size_t i = 0; i < pose_vec.size(); ++i)
      {
        if (curr_estimate.exists(gtsam::Symbol('b', i)))
        {
          pose_vec[i] = Eigen::Affine3d(curr_estimate.at<gtsam::Pose3>(gtsam::Symbol('b', i)).matrix());
        }
      }

      // Update T_est for TF (Feedback: T = Global_b0 * Local_b0_inv)
      // Since local b0 is Identity (at start), T = Global_b0.
      if (curr_estimate.exists(gtsam::Symbol('b', 0)))
      {
        T_W1_to_W0_est = pose_vec[0];
      }

      if (has_loop_flag)
      {
        PointCloud full_map;
        for (int i = 0; i < pose_vec.size(); ++i)
        {
          PointCloud correct_cloud;
          pcl::transformPointCloud(*cloud_vec[i], correct_cloud, pose_vec[i]);
          full_map += correct_cloud;
        }
        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(full_map, pub_cloud);
        pub_cloud.header.frame_id = W0_FRAME; // Published in W0
        pubCorrectCloud.publish(pub_cloud);

        nav_msgs::Path correct_path;
        correct_path.header.stamp = ros::Time::now();
        correct_path.header.frame_id = W0_FRAME;
        for (int i = 0; i < pose_vec.size(); i += 1)
        {
          geometry_msgs::PoseStamped msg_pose;
          msg_pose.pose.position.x = pose_vec[i].translation()[0];
          msg_pose.pose.position.y = pose_vec[i].translation()[1];
          msg_pose.pose.position.z = pose_vec[i].translation()[2];
          Eigen::Quaterniond pose_q(pose_vec[i].rotation());
          msg_pose.header.frame_id = W0_FRAME;
          msg_pose.pose.orientation.x = pose_q.x();
          msg_pose.pose.orientation.y = pose_q.y();
          msg_pose.pose.orientation.z = pose_q.z();
          msg_pose.pose.orientation.w = pose_q.w();
          correct_path.poses.push_back(msg_pose);
        }
        pubCorrectPath.publish(correct_path);
      }

      if (keyCloudInd % 10 == 0 && !path_s0.poses.empty())
      {
        path_s0.header.stamp = ros::Time::now();
        pubSession0Path.publish(path_s0);
      }

      visualizeLoopClosure(pubLoopConstraintEdge, loop_container, pose_vec);
      has_loop_flag = false;
      ++cloudInd;
    }
    else
    {
      // Save Logic
      ros::WallDuration no_data_duration = ros::WallTime::now() - last_data_time;
      if (!keyframes_saved && !keyframe_pose_vec.empty() && config_setting.keyframe_save_ && no_data_duration.toSec() > 5.0)
      {
        keyframes_saved = true;
        ROS_INFO("saving keyframe ...");
        boost::filesystem::create_directories(config_setting.pos_dir_);
        boost::filesystem::create_directories(config_setting.std_dir_);
        boost::filesystem::create_directories(config_setting.pcd_dir_);

        // Fix: Use separate string variable for saveDatabase
        string pose_file_name = config_setting.pos_dir_ + "poses.txt";
        string db_file_name = config_setting.std_dir_ + "std_database.txt";
        std_manager->saveDatabase(db_file_name);
        std::ofstream pose_file(pose_file_name);

        for (int i = 0; i < keyframe_pose_vec.size(); ++i)
        {
          std::ostringstream oss;
          oss << std::setw(4) << std::setfill('0') << i;
          string key_frame_idx = oss.str();
          pcl::io::savePCDFileBinary(config_setting.pcd_dir_ + key_frame_idx + ".pcd", *std_manager->key_cloud_vec_[i]);

          // Map keyframe index to global pose_vec index (assuming regular subsampling)
          int global_idx = i * config_setting.sub_frame_num_;
          if (global_idx < pose_vec.size())
          {
            Eigen::Quaterniond q(pose_vec[global_idx].rotation());
            q.normalize();
            auto t = pose_vec[global_idx].translation();
            pose_file << t.x() << ", " << t.y() << "," << t.z() << "," << q.w() << "," << q.x() << "," << q.y() << "," << q.z() << "\n";
          }
        }
        pose_file.close();
        ROS_INFO("saving done!");

        if (curr_estimate.exists(gtsam::Symbol('b', 0)))
        {
          T_W1_to_W0_est = Eigen::Affine3d(curr_estimate.at<gtsam::Pose3>(gtsam::Symbol('b', 0)).matrix());

          ROS_INFO("-------------------------------------------------------");
          ROS_INFO_STREAM("Final Estimated Transform (b_0 in W0): \n"
                          << T_W1_to_W0_est.matrix());
          ROS_INFO("-------------------------------------------------------");

          ROS_INFO("Saving aligned global map...");
          PointCloud final_global_map;

          for (size_t i = 0; i < cloud_vec.size(); ++i)
          {
            PointCloud tmp;
            pcl::transformPointCloud(*cloud_vec[i], tmp, pose_vec[i]); // Apply optimized Global Pose
            final_global_map += tmp;
          }

          for (const auto &cloud_ref : ref_keyframe_clouds)
          {
            final_global_map += *cloud_ref;
          }

          down_sampling_voxel(final_global_map, 0.05);
          std::string aligned_map_file = config_setting.pos_dir_ + "aligned/global_map_W0.pcd";
          boost::filesystem::create_directories(config_setting.pos_dir_ + "aligned/");
          pcl::io::savePCDFileBinary(aligned_map_file, final_global_map);
          ROS_INFO_STREAM("Aligned global map saved to: " << aligned_map_file);
        }
        ros::shutdown();
        break;
      }
    }
  }
  return 0;
}