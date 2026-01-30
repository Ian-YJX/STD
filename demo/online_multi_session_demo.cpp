/**
 * Online Multi-Session Demo (Decoupled, Session1-fixed)
 *  V 2.0
 * Requirements implemented:
 *  1) Load session0 from saved files (poses.txt + pcd/*.pcd) and build STD DB.
 *  2) Subscribe session1 topics and optimize session1 online (single-session backend, iSAM2).
 *  3) From an initial guess, refine inter-session transform T_W1_to_W0 using inter-session loop closures,
 *     with distance + consistency gates (const thresholds) and yaw-only update.
 *  4) Visualization:
 *       - session1 publishes in W1 frame (camera_init).
 *       - session0 publishes in W0 frame (map_s0), and TF (camera_init -> map_s0) makes session0 move in RViz.
 *  5) Saving via /save_map (rosservice Trigger):
 *       - Save session1 in the same format as session0 loader expects: cur_dir_/poses.txt and cur_dir_/pcd/*.pcd
 *       - Save one merged map (two sessions) in cur_dir_/aligned/merged_map_W1.pcd (in W1 frame).
 *
 * Notes:
 *  - Session0 is NOT inserted into the factor graph; session1 graph is independent.
 *  - Inter-session loop closures do NOT add factors into session1 graph; they only refine T_W1_to_W0.
 */

#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <boost/filesystem.hpp>
#include <std_srvs/Trigger.h>
#include <boost/function.hpp>

#include <mutex>
#include <queue>
#include <thread>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <tf2_ros/transform_broadcaster.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "../include/STDesc.h"
#include "../include/multi_session_util.h"

typedef pcl::PointXYZRGB PointType;
typedef pcl::PointCloud<PointType> PointCloud;

// ------------------------ Global buffers (sync) ------------------------
static std::mutex laser_mtx;
static std::mutex odom_mtx;

static std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
static std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;

static inline double wrapToPi(double a)
{
  while (a > M_PI)
    a -= 2.0 * M_PI;
  while (a < -M_PI)
    a += 2.0 * M_PI;
  return a;
}

static inline double yawFromRot(const Eigen::Matrix3d &R)
{
  return std::atan2(R(1, 0), R(0, 0));
}

static inline std::pair<double, double> poseDeltaDTDR(const Eigen::Affine3d &Td)
{
  double dt = Td.translation().norm();
  double dr = Eigen::AngleAxisd(Td.linear()).angle();
  return std::make_pair(dt, dr);
}

static inline Eigen::Affine3d chooseMeasClosestToPred(const Eigen::Affine3d &T_pred, const Eigen::Affine3d &T_icp)
{
  // some ICP implementations can return inverse direction; pick the closer one
  Eigen::Affine3d cand1 = T_icp;
  Eigen::Affine3d cand2 = T_icp.inverse();

  auto d1 = poseDeltaDTDR(T_pred.inverse() * cand1);
  auto d2 = poseDeltaDTDR(T_pred.inverse() * cand2);

  // weighted: translation + 2*rotation
  return (d2.first + 2.0 * d2.second < d1.first + 2.0 * d1.second) ? cand2 : cand1;
}

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

  if (std::fabs(odom_timestamp - laser_timestamp) < 1e-3)
  {
    pcl::fromROSMsg(*laser_msg, *cloud);

    Eigen::Quaterniond r(
        odom_msg->pose.pose.orientation.w,
        odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y,
        odom_msg->pose.pose.orientation.z);

    Eigen::Vector3d t(
        odom_msg->pose.pose.position.x,
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
    const std::vector<Eigen::Affine3d> &pose_vec,
    const std::string &frame_id)
{
  if (loop_container.empty())
    return;

  visualization_msgs::MarkerArray markerArray;

  visualization_msgs::Marker markerNode;
  markerNode.header.frame_id = frame_id;
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
  markerEdge.header.frame_id = frame_id;
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

  for (const auto &pr : loop_container)
  {
    int idx_cur = pr.first;
    int idx_pre = pr.second;
    if (idx_cur < 0 || idx_pre < 0)
      continue;
    if (idx_cur >= (int)pose_vec.size() || idx_pre >= (int)pose_vec.size())
      continue;

    geometry_msgs::Point p;
    p.x = pose_vec[idx_cur].translation().x();
    p.y = pose_vec[idx_cur].translation().y();
    p.z = pose_vec[idx_cur].translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);

    p.x = pose_vec[idx_pre].translation().x();
    p.y = pose_vec[idx_pre].translation().y();
    p.z = pose_vec[idx_pre].translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
  }

  markerArray.markers.push_back(markerNode);
  markerArray.markers.push_back(markerEdge);
  publisher.publish(markerArray);
}

static bool ensureDir(const std::string &dir)
{
  try
  {
    boost::filesystem::create_directories(dir);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "online_multi_session_demo");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  ConfigSetting config_setting;
  read_parameters(nh, config_setting);

  // Frames: session1 fixed in W1, session0 in W0
  static const std::string W1_FRAME = "camera_init"; // session1 fixed
  static const std::string W0_FRAME = "map_s0";      // session0 moves via TF (child of camera_init)

  // ------------------------ Publishers ------------------------
  ros::Publisher pubOriginCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_origin", 10000);
  ros::Publisher pubCorrectCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 10000);
  ros::Publisher pubCorrectPath = nh.advertise<nav_msgs::Path>("/correct_path", 100000);
  ros::Publisher pubOdomOrigin = nh.advertise<nav_msgs::Odometry>("/odom_origin", 10);

  ros::Publisher pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 10);
  ros::Publisher pubMSLoopEdge = nh.advertise<visualization_msgs::MarkerArray>("/multi_session_loop", 10);

  ros::Publisher pubSession0Cloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_session0", 1, true);
  ros::Publisher pubSession0Path = nh.advertise<nav_msgs::Path>("/path_session0", 1, true);

  // Debug publishers (optional)
  ros::Publisher pubCurrentCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
  ros::Publisher pubSTD = nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

  // ------------------------ Subscribers ------------------------
  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered_body", 100, laserCloudHandler);
  ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, OdomHandler);

  // ------------------------ TF broadcaster ------------------------
  tf2_ros::TransformBroadcaster tf_br;

  // ------------------------ Managers ------------------------
  std::shared_ptr<STDescManager> std_manager_s1(new STDescManager(config_setting));    // session1 DB (intra-session)
  std::shared_ptr<STDescManager> std_manager_prior(new STDescManager(config_setting)); // session0 DB + planes

  // ------------------------ Factor graph for session1 only ------------------------
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;

  auto odometryNoise = gtsam::noiseModel::Diagonal::Variances(
      (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

  // session1 start prior: loose
  auto s1StartPriorNoise = gtsam::noiseModel::Diagonal::Variances(
      (gtsam::Vector(6) << 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0).finished());

  double intraScore = 0.01;
  gtsam::Vector robustNoiseVector6(6);
  robustNoiseVector6 << intraScore, intraScore, intraScore, intraScore, intraScore, intraScore;
  auto robustLoopNoise = gtsam::noiseModel::Robust::Create(
      gtsam::noiseModel::mEstimator::Cauchy::Create(1),
      gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  gtsam::ISAM2 isam(parameters);

  // ------------------------ Load / init T_W1_to_W0 ------------------------
  Eigen::Affine3d T_W1_to_W0_est = Eigen::Affine3d::Identity();
  std::vector<double> matrix_vals;
  if (nh.getParam("multi_session/initial_T_W1_W0", matrix_vals) && matrix_vals.size() == 16)
  {
    Eigen::Matrix4d mat;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        mat(i, j) = matrix_vals[i * 4 + j];
    T_W1_to_W0_est = Eigen::Affine3d(mat);
    ROS_INFO("Loaded initial T_W1_to_W0 from ROS params.");
  }
  else
  {
    ROS_WARN("Failed to load multi_session/initial_T_W1_W0, using identity.");
  }

  // ------------------------ Multi-session gating & T update params (const) ------------------------
  double inter_session_max_dist = 30.0; // meters (distance gate in W0)
  double inter_consis_max_trans = 5.0;  // meters
  double inter_consis_max_rot = 0.52;   // rad (~30 deg)

  // T update parameters
  double t_update_alpha = 0.15;    // [0,1], damping for T updates
  int t_update_max_iters = 5;      // estimateSessionTransform iters per update
  int t_update_window = 30;        // keep last N inter-session loops
  double t_stuck_eps_trans = 1e-4; // meters
  double t_stuck_eps_yaw = 1e-4;   // rad
  int t_stuck_K = 5;               // consecutive times
  double correct_map_ds = 0.2;     // voxel size for /cloud_correct
  double merged_map_ds = 0.05;     // voxel size for merged_map saving

  nh.param("multi_session/inter_session_max_dist", inter_session_max_dist, inter_session_max_dist);
  nh.param("multi_session/inter_consis_max_trans", inter_consis_max_trans, inter_consis_max_trans);
  nh.param("multi_session/inter_consis_max_rot", inter_consis_max_rot, inter_consis_max_rot);

  nh.param("multi_session/t_update_alpha", t_update_alpha, t_update_alpha);
  nh.param("multi_session/t_update_max_iters", t_update_max_iters, t_update_max_iters);
  nh.param("multi_session/t_update_window", t_update_window, t_update_window);

  nh.param("multi_session/t_stuck_eps_trans", t_stuck_eps_trans, t_stuck_eps_trans);
  nh.param("multi_session/t_stuck_eps_yaw", t_stuck_eps_yaw, t_stuck_eps_yaw);
  nh.param("multi_session/t_stuck_K", t_stuck_K, t_stuck_K);

  nh.param("multi_session/correct_map_ds", correct_map_ds, correct_map_ds);
  nh.param("multi_session/merged_map_ds", merged_map_ds, merged_map_ds);

  // ------------------------ Runtime buffers ------------------------
  size_t cloudInd = 0;
  size_t keyCloudInd = 0;

  std::vector<PointCloud::Ptr> cloud_vec_body;     // per-frame clouds in body
  std::vector<Eigen::Affine3d> pose_vec_W1;        // per-frame optimized poses in W1
  std::vector<Eigen::Affine3d> odom_pose_vec_W1;   // per-frame raw odom poses (W1)
  std::vector<size_t> keyframe_node_idx;           // keyframe index -> node index
  std::vector<std::pair<int, int>> loop_container; // (node_cur, node_match) for intra loops

  // session0 cache
  std::vector<Eigen::Affine3d> ref_keyframe_poses_W0;
  std::vector<PointCloud::Ptr> ref_keyframe_clouds_W0;
  nav_msgs::Path path_s0;

  // inter-session loops window (stores T observations)
  std::vector<InterSessionLoop> inter_loops_window;

  // misc
  std::mutex data_mtx;
  gtsam::Values curr_estimate;

  // keyframe accumulating cloud (in W1)
  PointCloud::Ptr key_cloud_W1(new PointCloud);

  // ------------------------ Load session0 (poses + pcd + STD DB) ------------------------
  if (config_setting.multi_session_mode_ == 1)
  {
    if (!loadSessionAndBuildSTD(ref_keyframe_poses_W0, ref_keyframe_clouds_W0, config_setting, 0, std_manager_prior))
    {
      ROS_ERROR("Failed to load reference session0.");
    }
    else
    {
      ROS_INFO("Session0 loaded: keyframes=%zu, db=%zu", ref_keyframe_poses_W0.size(), std_manager_prior->data_base_.size());

      // Publish merged session0 map (latched), in W0_FRAME
      PointCloud::Ptr full_s0(new PointCloud);
      for (auto &c : ref_keyframe_clouds_W0)
      {
        if (c && !c->empty())
          *full_s0 += *c;
      }
      if (!full_s0->empty())
      {
        down_sampling_voxel(*full_s0, 0.2);
        sensor_msgs::PointCloud2 map_msg;
        pcl::toROSMsg(*full_s0, map_msg);
        map_msg.header.frame_id = W0_FRAME;
        map_msg.header.stamp = ros::Time::now();
        pubSession0Cloud.publish(map_msg);
        ROS_INFO("Published merged session0 map (W0) size=%zu", full_s0->size());
      }

      // Publish session0 path (latched), in W0_FRAME
      path_s0.header.frame_id = W0_FRAME;
      path_s0.header.stamp = ros::Time::now();
      for (const auto &pose : ref_keyframe_poses_W0)
      {
        geometry_msgs::PoseStamped ps;
        ps.header.frame_id = W0_FRAME;
        ps.pose.position.x = pose.translation().x();
        ps.pose.position.y = pose.translation().y();
        ps.pose.position.z = pose.translation().z();
        Eigen::Quaterniond q(pose.rotation());
        q.normalize();
        ps.pose.orientation.w = q.w();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();
        path_s0.poses.push_back(ps);
      }
      pubSession0Path.publish(path_s0);
      ROS_INFO("Published session0 path (W0).");
    }
  }

  // ------------------------ /save_map service (save session1 + merged map) ------------------------
  bool keyframes_saved = false;

  auto saveMapLambda = [&](std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res) -> bool
  {
    std::unique_lock<std::mutex> lk(data_mtx);

    if (keyframes_saved)
    {
      res.success = false;
      res.message = "Map already saved.";
      ROS_WARN_STREAM(res.message);
      return true;
    }
    if (keyframe_node_idx.empty() || std_manager_s1->key_cloud_vec_.empty())
    {
      res.success = false;
      res.message = "No session1 keyframes to save.";
      ROS_WARN_STREAM(res.message);
      return true;
    }

    const std::string out_root = config_setting.cur_dir_; // to match loader expectation
    const std::string out_pcd_dir = out_root + "pcd/";
    const std::string out_pose_file = out_root + "poses.txt";
    const std::string out_aligned_dir = out_root + "aligned/";
    const std::string out_merged_map = out_aligned_dir + "merged_map_W1.pcd";

    if (!ensureDir(out_pcd_dir) || !ensureDir(out_aligned_dir))
    {
      res.success = false;
      res.message = "Failed to create output directories under cur_dir_.";
      ROS_ERROR_STREAM(res.message);
      return true;
    }

    ROS_INFO("Saving session1 keyframes to %s", out_root.c_str());

    // Save PCDs + poses.txt
    std::ofstream pose_file(out_pose_file);
    if (!pose_file.is_open())
    {
      res.success = false;
      res.message = "Cannot open poses.txt for writing.";
      ROS_ERROR_STREAM(res.message << " file=" << out_pose_file);
      return true;
    }

    const size_t K = std::min(keyframe_node_idx.size(), std_manager_s1->key_cloud_vec_.size());
    for (size_t i = 0; i < K; ++i)
    {
      // pcd name: 0000.pcd ...
      std::ostringstream oss;
      oss << std::setw(4) << std::setfill('0') << i;
      const std::string idx_str = oss.str();
      const std::string pcd_path = out_pcd_dir + idx_str + ".pcd";

      // key cloud in W1
      pcl::io::savePCDFileBinary(pcd_path, *std_manager_s1->key_cloud_vec_[i]);

      const size_t node_idx = keyframe_node_idx[i];
      if (node_idx >= pose_vec_W1.size())
        continue;

      Eigen::Quaterniond q(pose_vec_W1[node_idx].rotation());
      q.normalize();
      Eigen::Vector3d t = pose_vec_W1[node_idx].translation();

      // format: x, y, z, qw, qx, qy, qz
      pose_file << t.x() << ", " << t.y() << ", " << t.z() << ", "
                << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "\n";
    }
    pose_file.close();

    // Save merged map in W1: (session1 key clouds) + (session0 clouds transformed to W1)
    ROS_INFO("Saving merged two-session map to %s", out_merged_map.c_str());
    PointCloud merged;

    // session1: merge key clouds (already in W1)
    for (size_t i = 0; i < K; ++i)
    {
      merged += *std_manager_s1->key_cloud_vec_[i];
    }

    // session0: transform from W0 -> W1 using current estimate
    Eigen::Affine3d T_W0_to_W1 = T_W1_to_W0_est.inverse();
    for (const auto &c0 : ref_keyframe_clouds_W0)
    {
      if (!c0 || c0->empty())
        continue;
      PointCloud tmp;
      pcl::transformPointCloud(*c0, tmp, T_W0_to_W1);
      merged += tmp;
    }

    if (merged_map_ds > 1e-6)
      down_sampling_voxel(merged, merged_map_ds);

    pcl::io::savePCDFileBinary(out_merged_map, merged);

    keyframes_saved = true;
    res.success = true;
    res.message = "Saved session1 (poses.txt + pcd/) and merged_map_W1.pcd.";
    ROS_INFO_STREAM(res.message);
    return true;
  };
  ros::ServiceServer save_map_service =
      nh.advertiseService("/save_map",
                          boost::function<bool(std_srvs::Trigger::Request &, std_srvs::Trigger::Response &)>(saveMapLambda));

  // ros::ServiceServer save_map_service = nh.advertiseService("/save_map", saveMapLambda);
  ROS_INFO("Service /save_map is ready.");

  // ------------------------ Spinner / main loop ------------------------
  ros::AsyncSpinner spinner(2);
  spinner.start();
  ros::WallRate rate(100.0);

  int t_stuck_cnt = 0;
  std::ofstream loop_log_file;
  loop_log_file.open(config_setting.cur_dir_ + "inter_session_loops.log");
  if (loop_log_file.is_open())
    loop_log_file << "# ros_time, s1_kf, s0_kf, score, dist_w0\n";

  while (ros::ok())
  {
    PointCloud::Ptr current_cloud_body(new PointCloud);
    Eigen::Affine3d pose_odom_W1;

    if (!syncPackages(current_cloud_body, pose_odom_W1))
    {
      rate.sleep();
      continue;
    }

    // Build current cloud in W1 (camera_init)
    PointCloud::Ptr current_cloud_W1(new PointCloud);
    pcl::transformPointCloud(*current_cloud_body, *current_cloud_W1, pose_odom_W1);
    down_sampling_voxel(*current_cloud_W1, config_setting.ds_size_);

    // publish origin cloud (W1)
    sensor_msgs::PointCloud2 origin_msg;
    pcl::toROSMsg(*current_cloud_W1, origin_msg);
    origin_msg.header.frame_id = W1_FRAME;
    origin_msg.header.stamp = ros::Time::now();
    pubOriginCloud.publish(origin_msg);

    // publish odom (W1)
    Eigen::Quaterniond q_odom(pose_odom_W1.rotation());
    q_odom.normalize();
    nav_msgs::Odometry odom_msg;
    odom_msg.header.frame_id = W1_FRAME;
    odom_msg.header.stamp = origin_msg.header.stamp;
    odom_msg.pose.pose.position.x = pose_odom_W1.translation().x();
    odom_msg.pose.pose.position.y = pose_odom_W1.translation().y();
    odom_msg.pose.pose.position.z = pose_odom_W1.translation().z();
    odom_msg.pose.pose.orientation.w = q_odom.w();
    odom_msg.pose.pose.orientation.x = q_odom.x();
    odom_msg.pose.pose.orientation.y = q_odom.y();
    odom_msg.pose.pose.orientation.z = q_odom.z();
    pubOdomOrigin.publish(odom_msg);

    // push runtime buffers
    {
      std::unique_lock<std::mutex> lk(data_mtx);
      cloud_vec_body.push_back(current_cloud_body);
      odom_pose_vec_W1.push_back(pose_odom_W1);

      // store initial guess for current node in pose_vec_W1
      pose_vec_W1.push_back(pose_odom_W1);
    }

    // Insert into graph (session1 only)
    initial.insert(gtsam::Symbol('b', cloudInd), gtsam::Pose3(pose_odom_W1.matrix()));

    if (cloudInd == 0)
    {
      graph.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Symbol('b', 0), gtsam::Pose3(pose_odom_W1.matrix()), s1StartPriorNoise));
    }
    else
    {
      gtsam::Pose3 prev_pose(odom_pose_vec_W1[cloudInd - 1].matrix());
      gtsam::Pose3 curr_pose(pose_odom_W1.matrix());
      graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
          gtsam::Symbol('b', cloudInd - 1), gtsam::Symbol('b', cloudInd),
          prev_pose.between(curr_pose), odometryNoise));
    }

    // accumulate key cloud (W1)
    *key_cloud_W1 += *current_cloud_W1;

    // pending inter-loop measurement for this keyframe (computed before optimization, applied after)
    bool pending_inter_loop = false;
    int pending_ref_kf = -1;
    int pending_cur_kf = -1;
    size_t pending_cur_node = 0;
    double pending_score = 0.0;
    double pending_dist_w0 = 0.0;
    Eigen::Affine3d pending_T_meas_W0 = Eigen::Affine3d::Identity(); // ref -> cur in W0

    bool added_any_loop_factor = false;

    // ------------------------ Keyframe event ------------------------
    if (cloudInd % config_setting.sub_frame_num_ == 0 && cloudInd != 0)
    {
      const size_t cur_node = cloudInd;
      const int cur_kf = static_cast<int>(keyCloudInd);
      keyframe_node_idx.push_back(cur_node);

      ROS_INFO("Keyframe: kf=%d node=b%zu", cur_kf, cur_node);

      // Debug publish key cloud (W1)
      sensor_msgs::PointCloud2 key_msg;
      pcl::toROSMsg(*key_cloud_W1, key_msg);
      key_msg.header.frame_id = W1_FRAME;
      key_msg.header.stamp = ros::Time::now();
      pubCurrentCloud.publish(key_msg);

      // Generate STDescs for session1 (intra-session DB, W1)
      std::vector<STDesc> stds_s1;
      std_manager_s1->GenerateSTDescs(key_cloud_W1, stds_s1);

      // store key cloud for saving (W1)
      std_manager_s1->key_cloud_vec_.push_back(key_cloud_W1->makeShared());

      // ---------------- Intra-session loop closure (session1) ----------------
      if (cur_kf > config_setting.skip_near_num_)
      {
        std::pair<int, double> search_result(-1, 0.0);
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        loop_transform.first.setZero();
        loop_transform.second.setIdentity();
        std::vector<std::pair<STDesc, STDesc>> loop_std_pair;

        std_manager_s1->SearchLoop(stds_s1, search_result, loop_transform, loop_std_pair, std_manager_s1->data_base_);

        if (search_result.first >= 0)
        {
          const int match_kf = search_result.first;
          if (match_kf >= 0 && match_kf < (int)keyframe_node_idx.size())
          {
            const size_t match_node = keyframe_node_idx[match_kf];

            // ICP refine: query=current, ref=match (both in W1)
            std_manager_s1->PlaneGeometricIcp(std_manager_s1->plane_cloud_vec_.back(),
                                              std_manager_s1->plane_cloud_vec_[match_kf],
                                              loop_transform);

            Eigen::Affine3d T_icp = Eigen::Affine3d::Identity();
            T_icp.linear() = loop_transform.second;
            T_icp.translation() = loop_transform.first;

            // predicted relative (match -> current) in W1 (odom quick prediction)
            Eigen::Affine3d T_pred = odom_pose_vec_W1[match_node].inverse() * odom_pose_vec_W1[cur_node];
            Eigen::Affine3d T_meas = chooseMeasClosestToPred(T_pred, T_icp);

            // Following online_demo.cpp: apply delta_T (current->match) to refine source poses,
            // then add robust constraints between corresponding subframes.
            Eigen::Affine3d delta_T = T_meas.inverse(); // current->match

            int sub_frame_num = config_setting.sub_frame_num_;
            for (int j = 1; j <= sub_frame_num; ++j)
            {
              int src_frame = static_cast<int>(cur_node) + j - sub_frame_num;   // in current keyframe window
              int tar_frame = static_cast<int>(match_node) + j - sub_frame_num; // in matched keyframe window
              if (src_frame < 0 || tar_frame < 0)
                continue;
              if (src_frame >= static_cast<int>(odom_pose_vec_W1.size()) || tar_frame >= static_cast<int>(odom_pose_vec_W1.size()))
                continue;

              Eigen::Affine3d src_pose_refined = delta_T * odom_pose_vec_W1[src_frame]; // refined in target-aligned world
              Eigen::Affine3d tar_pose = odom_pose_vec_W1[tar_frame];

              gtsam::Pose3 meas = gtsam::Pose3(tar_pose.matrix()).between(gtsam::Pose3(src_pose_refined.matrix()));
              graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                  gtsam::Symbol('b', tar_frame), gtsam::Symbol('b', src_frame),
                  meas, robustLoopNoise));

              loop_container.emplace_back(src_frame, tar_frame);
            }

            // cap visualization edges to avoid RViz overload
            if (loop_container.size() > 2000)
              loop_container.erase(loop_container.begin(), loop_container.begin() + (loop_container.size() - 2000));

            added_any_loop_factor = true;

            publish_std_pairs(loop_std_pair, pubSTD);
            ROS_INFO("[Intra Loop] b%zu <-> b%zu (kf %d <-> %d)", cur_node, match_node, cur_kf, match_kf);
          }
        }
      }

      // add current stds to session1 DB
      // ensure frame_id_ is keyframe index for DB consistency
      for (auto &d : stds_s1)
        d.frame_id_ = cur_kf;
      std_manager_s1->AddSTDescs(stds_s1);

      // ---------------- Inter-session loop closure (S1 keyframe -> S0 DB) ----------------
      if (config_setting.multi_session_mode_ == 1 && !ref_keyframe_poses_W0.empty())
      {
        // Build query key cloud in W0 using current T estimate
        PointCloud::Ptr key_cloud_W0_guess(new PointCloud);
        pcl::transformPointCloud(*key_cloud_W1, *key_cloud_W0_guess, T_W1_to_W0_est);

        // IMPORTANT: Generate query descriptors on std_manager_prior (so SearchLoop uses plane_cloud_vec_.back() as query)
        std::vector<STDesc> stds_q;
        std_manager_prior->GenerateSTDescs(key_cloud_W0_guess, stds_q);
        if (!stds_q.empty())
        {
          const int query_plane_index = static_cast<int>(std_manager_prior->plane_cloud_vec_.size()) - 1;
          std_manager_prior->current_frame_id_ = query_plane_index;

          // bypass skip_near_num_ for inter-session by setting huge frame_id
          const int LARGE_FRAME_ID_BASE = 1000000;
          for (auto &d : stds_q)
            d.frame_id_ = LARGE_FRAME_ID_BASE + cur_kf;

          std::pair<int, double> ms_result(-1, 0.0);
          std::pair<Eigen::Vector3d, Eigen::Matrix3d> ms_loop_transform;
          ms_loop_transform.first.setZero();
          ms_loop_transform.second.setIdentity();
          std::vector<std::pair<STDesc, STDesc>> ms_loop_std_pair;

          std_manager_prior->SearchLoop(stds_q, ms_result, ms_loop_transform, ms_loop_std_pair, std_manager_prior->data_base_, 1);

          const int ms_match_kf = ms_result.first;
          const double ms_score = ms_result.second;

          if (ms_match_kf >= 0 && ms_score > config_setting.inter_session_icp_threshold_ && ms_match_kf < (int)ref_keyframe_poses_W0.size())
          {
            // distance gate in W0: compare predicted S1 pose in W0 to matched S0 pose in W0
            Eigen::Affine3d B_guess_W1 = odom_pose_vec_W1[cur_node];
            Eigen::Affine3d B_in_W0_guess = T_W1_to_W0_est * B_guess_W1;

            Eigen::Vector3d p_s1_w0 = B_in_W0_guess.translation();
            Eigen::Vector3d p_s0_w0 = ref_keyframe_poses_W0[ms_match_kf].translation();
            double dist_w0 = (p_s1_w0 - p_s0_w0).norm();

            // ICP refine: query plane is last (back) in prior manager, ref plane is ms_match_kf
            std_manager_prior->PlaneGeometricIcp(std_manager_prior->plane_cloud_vec_.back(),
                                                 std_manager_prior->plane_cloud_vec_[ms_match_kf],
                                                 ms_loop_transform);

            Eigen::Affine3d T_icp = Eigen::Affine3d::Identity();
            T_icp.linear() = ms_loop_transform.second;
            T_icp.translation() = ms_loop_transform.first;

            // predicted relative (ref -> cur) in W0
            Eigen::Affine3d T_pred = ref_keyframe_poses_W0[ms_match_kf].inverse() * B_in_W0_guess;
            Eigen::Affine3d T_meas = chooseMeasClosestToPred(T_pred, T_icp);

            // consistency gate
            auto dc = poseDeltaDTDR(T_pred.inverse() * T_meas);
            if (dc.first <= inter_consis_max_trans && dc.second <= inter_consis_max_rot)
            {
              pending_inter_loop = true;
              pending_ref_kf = ms_match_kf;
              pending_cur_kf = cur_kf;
              pending_cur_node = cur_node;
              pending_score = ms_score;
              pending_dist_w0 = dist_w0;
              pending_T_meas_W0 = T_meas;

              publish_std_pairs(ms_loop_std_pair, pubSTD);

              ROS_INFO("[Inter Loop Accepted] s1_kf=%d node=b%zu <-> s0_kf=%d score=%.4f dist=%.2f d(dt=%.2f dr=%.2f)",
                       cur_kf, cur_node, ms_match_kf, ms_score, dist_w0, dc.first, dc.second);

              if (loop_log_file.is_open())
                loop_log_file << ros::Time::now() << ", " << cur_kf << ", " << ms_match_kf << ", " << ms_score << ", " << dist_w0 << "\n";
              added_any_loop_factor = true;
            }
            else
            {
              ROS_WARN("[Inter Loop Rejected: consistency] dt=%.2f dr=%.2f score=%.4f dist=%.2f",
                       dc.first, dc.second, ms_score, dist_w0);
            }
          }
        }
      }

      // reset key cloud accumulator
      key_cloud_W1->clear();
      ++keyCloudInd;
    }

    // ------------------------ Optimize session1 ------------------------
    isam.update(graph, initial);
    isam.update();
    if (added_any_loop_factor)
    {
      for (int k = 0; k < 5; ++k)
        isam.update();
    }
    graph.resize(0);
    initial.clear();

    curr_estimate = isam.calculateEstimate();

    // update pose_vec_W1 from estimate
    {
      std::unique_lock<std::mutex> lk(data_mtx);
      for (size_t i = 0; i < pose_vec_W1.size(); ++i)
      {
        if (curr_estimate.exists(gtsam::Symbol('b', i)))
          pose_vec_W1[i] = Eigen::Affine3d(curr_estimate.at<gtsam::Pose3>(gtsam::Symbol('b', i)).matrix());
      }
    }

    // ------------------------ Update T (W1->W0) ONLY when inter-session loop accepted ------------------------
    if (pending_inter_loop && pending_ref_kf >= 0 && pending_ref_kf < (int)ref_keyframe_poses_W0.size())
    {
      std::unique_lock<std::mutex> lk(data_mtx);

      if (pending_cur_node < pose_vec_W1.size())
      {
        const Eigen::Affine3d &A_W0 = ref_keyframe_poses_W0[pending_ref_kf];
        const Eigen::Affine3d &B_W1 = pose_vec_W1[pending_cur_node]; // optimized

        // Observation of T_W1_to_W0:
        //   A * (ref->cur meas in W0) * inv(B)  =  T_W1_to_W0
        Eigen::Affine3d T_obs = A_W0 * pending_T_meas_W0 * B_W1.inverse();

        // Fill an InterSessionLoop item as an observation of T
        InterSessionLoop obs;
        obs.ref_kf = pending_ref_kf;
        obs.cur_kf = pending_cur_kf;
        obs.score = pending_score;
        obs.relative_pose.first = T_obs.translation();
        obs.relative_pose.second = T_obs.linear();
        inter_loops_window.push_back(obs);
        if ((int)inter_loops_window.size() > t_update_window)
          inter_loops_window.erase(inter_loops_window.begin(), inter_loops_window.begin() + (inter_loops_window.size() - t_update_window));

        Eigen::Affine3d T_before = T_W1_to_W0_est;

        Eigen::Affine3d T_refined = estimateSessionTransform(inter_loops_window, T_W1_to_W0_est, true, t_update_max_iters);

        // damping update
        double alpha = std::min(1.0, std::max(0.0, t_update_alpha));
        Eigen::Quaterniond q_old(T_W1_to_W0_est.rotation());
        Eigen::Quaterniond q_new(T_refined.rotation());
        q_old.normalize();
        q_new.normalize();
        Eigen::Quaterniond q_blend = q_old.slerp(alpha, q_new);
        q_blend.normalize();

        Eigen::Vector3d t_old = T_W1_to_W0_est.translation();
        Eigen::Vector3d t_blend = (1.0 - alpha) * t_old + alpha * T_refined.translation();

        T_W1_to_W0_est = Eigen::Affine3d::Identity();
        T_W1_to_W0_est.linear() = q_blend.toRotationMatrix();
        T_W1_to_W0_est.translation() = t_blend;

        // stuck detection
        double dt = (T_W1_to_W0_est.translation() - T_before.translation()).norm();
        double dyaw = wrapToPi(yawFromRot(T_W1_to_W0_est.rotation()) - yawFromRot(T_before.rotation()));
        if (dt < t_stuck_eps_trans && std::fabs(dyaw) < t_stuck_eps_yaw)
          t_stuck_cnt++;
        else
          t_stuck_cnt = 0;

        if (t_stuck_cnt >= t_stuck_K)
        {
          ROS_INFO("[T Update] appears stuck (cnt=%d). dt=%.3e dyaw=%.3e. Boosting alpha temporarily.",
                   t_stuck_cnt, dt, dyaw);

          // boost alpha once
          double alpha_boost = std::min(1.0, std::max(alpha, std::min(1.0, alpha * 2.0)));
          Eigen::Quaterniond qb = q_old.slerp(alpha_boost, q_new);
          qb.normalize();
          Eigen::Vector3d tb = (1.0 - alpha_boost) * t_old + alpha_boost * T_refined.translation();

          Eigen::Affine3d T_boost = Eigen::Affine3d::Identity();
          T_boost.linear() = qb.toRotationMatrix();
          T_boost.translation() = tb;

          T_W1_to_W0_est = T_boost;
          t_stuck_cnt = 0; // reset after action
        }

        ROS_INFO_STREAM("[T Updated] window=" << inter_loops_window.size()
                                              << " alpha=" << alpha
                                              << " T_W1_to_W0:\n"
                                              << T_W1_to_W0_est.matrix());

        // visualize inter-session loop in W1: show B (W1) and A transformed to W1
        Eigen::Affine3d T_W0_to_W1 = T_W1_to_W0_est.inverse();
        Eigen::Affine3d A_in_W1 = T_W0_to_W1 * A_W0;
        visualizeMultiSessionLoop(pubMSLoopEdge, B_W1, A_in_W1);
      }
    }

    // ------------------------ Publish TF (camera_init -> map_s0) ------------------------
    {
      Eigen::Affine3d T_W0_to_W1 = T_W1_to_W0_est.inverse();

      geometry_msgs::TransformStamped ts;
      ts.header.stamp = ros::Time::now();
      ts.header.frame_id = W1_FRAME; // parent fixed
      ts.child_frame_id = W0_FRAME;  // child moves

      Eigen::Quaterniond q(T_W0_to_W1.rotation());
      q.normalize();
      ts.transform.translation.x = T_W0_to_W1.translation().x();
      ts.transform.translation.y = T_W0_to_W1.translation().y();
      ts.transform.translation.z = T_W0_to_W1.translation().z();
      ts.transform.rotation.w = q.w();
      ts.transform.rotation.x = q.x();
      ts.transform.rotation.y = q.y();
      ts.transform.rotation.z = q.z();
      tf_br.sendTransform(ts);
    }

    // ------------------------ Publish optimized session1 path (W1) ------------------------
    nav_msgs::Path path_s1_opt;
    path_s1_opt.header.stamp = ros::Time::now();
    path_s1_opt.header.frame_id = W1_FRAME;
    {
      std::unique_lock<std::mutex> lk(data_mtx);
      path_s1_opt.poses.reserve(pose_vec_W1.size());
      for (size_t i = 0; i < pose_vec_W1.size(); ++i)
      {
        geometry_msgs::PoseStamped ps;
        ps.header.stamp = path_s1_opt.header.stamp;
        ps.header.frame_id = W1_FRAME;

        ps.pose.position.x = pose_vec_W1[i].translation().x();
        ps.pose.position.y = pose_vec_W1[i].translation().y();
        ps.pose.position.z = pose_vec_W1[i].translation().z();

        Eigen::Quaterniond q(pose_vec_W1[i].rotation());
        q.normalize();
        ps.pose.orientation.w = q.w();
        ps.pose.orientation.x = q.x();
        ps.pose.orientation.y = q.y();
        ps.pose.orientation.z = q.z();

        path_s1_opt.poses.push_back(ps);
      }
    }
    pubCorrectPath.publish(path_s1_opt);

    // ------------------------ Publish corrected map (W1) when loop factor was added ------------------------
    if (added_any_loop_factor)
    {
      PointCloud full_map;
      {
        std::unique_lock<std::mutex> lk(data_mtx);
        for (size_t i = 0; i < pose_vec_W1.size(); ++i)
        {
          if (!cloud_vec_body[i] || cloud_vec_body[i]->empty())
            continue;
          PointCloud tmp;
          pcl::transformPointCloud(*cloud_vec_body[i], tmp, pose_vec_W1[i]);
          full_map += tmp;
        }
      }

      if (correct_map_ds > 1e-6)
        down_sampling_voxel(full_map, correct_map_ds);

      sensor_msgs::PointCloud2 map_msg;
      pcl::toROSMsg(full_map, map_msg);
      map_msg.header.frame_id = W1_FRAME;
      map_msg.header.stamp = ros::Time::now();
      pubCorrectCloud.publish(map_msg);
    }

    // publish session0 path occasionally (latched anyway)
    if (!path_s0.poses.empty() && (cloudInd % 500 == 0))
    {
      path_s0.header.stamp = ros::Time::now();
      pubSession0Path.publish(path_s0);
    }

    // visualize intra loops in W1
    visualizeLoopClosure(pubLoopConstraintEdge, loop_container, pose_vec_W1, W1_FRAME);

    ++cloudInd;
    rate.sleep();
  }

  return 0;
}