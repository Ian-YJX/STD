#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
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

#include "../include/STDesc.h"
#include "../include/multi_session_util.h"
#include "ros/init.h"

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;

// Global variables for multi-session
// int multisession_mode = 0;
// std::string save_directory;
// std::vector<Eigen::Affine3d> keyframe_poses_prior;
// std::vector<PointCloud::Ptr> keyframe_clouds_prior;

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

  // check if timestamps are matched
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
    ROS_WARN("Current odometry is earlier than laser scan, discard one "
             "odometry data.");
    std::unique_lock<std::mutex> o_lock(odom_mtx);
    odom_buffer.pop();
    return false;
  }
  else
  {
    ROS_WARN(
        "Current laser scan is earlier than odometry, discard one laser scan.");
    std::unique_lock<std::mutex> l_lock(laser_mtx);
    laser_buffer.pop();
    return false;
  }

  return true;
}

void update_poses(const gtsam::Values &estimates,
                  std::vector<Eigen::Affine3d> &poses)
{
  assert(estimates.size() == poses.size());

  poses.clear();

  for (int i = 0; i < estimates.size(); ++i)
  {
    auto est = estimates.at<gtsam::Pose3>(i);
    Eigen::Affine3d est_affine3d(est.matrix());
    poses.push_back(est_affine3d);
  }
}

void visualizeLoopClosure(
    const ros::Publisher &publisher,
    const std::vector<std::pair<int, int>> &loop_container,
    const std::vector<Eigen::Affine3d> &key_pose_vec)
{
  if (loop_container.empty())
    return;

  visualization_msgs::MarkerArray markerArray;
  // 闭环顶点
  visualization_msgs::Marker markerNode;
  markerNode.header.frame_id = "camera_init"; // camera_init
  // markerNode.header.stamp = ros::Time().fromSec( keyframeTimes.back() );
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
  // 闭环边
  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "camera_init";
  // markerEdge.header.stamp = ros::Time().fromSec( keyframeTimes.back() );
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

  // 遍历闭环
  for (auto it = loop_container.begin(); it != loop_container.end(); ++it)
  {
    int key_cur = it->first;
    int key_pre = it->second;
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

  ConfigSetting config_setting;
  read_parameters(nh, config_setting);

  // ros::Publisher pubOdomAftMapped =
  // nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  ros::Publisher pubCurrentCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
  ros::Publisher pubCurrentCorner =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100); // key
  ros::Publisher pubMatchedCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
  ros::Publisher pubMatchedCorner =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100); // key
  ros::Publisher pubSTD =
      nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10); // key

  ros::Publisher pubOriginCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_origin", 10000);

  ros::Publisher pubCorrectCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 10000);
  ros::Publisher pubCorrectPath =
      nh.advertise<nav_msgs::Path>("/correct_path", 100000);

  ros::Publisher pubOdomOrigin =
      nh.advertise<nav_msgs::Odometry>("/odom_origin", 10);
  ros::Publisher pubLoopConstraintEdge =
      nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints",
                                                    10); // key
  ros::Publisher pubMSLoopConstraintEdge =
      nh.advertise<visualization_msgs::MarkerArray>("/ms_loop_closure_constraints", 10);
  ros::Publisher pubRefMap =
      nh.advertise<sensor_msgs::PointCloud2>("/ref_map", 1, true);
  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
      "/cloud_registered_body", 100, laserCloudHandler);
  ros::Subscriber subOdom =
      nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, OdomHandler);

  STDescManager *std_manager = new STDescManager(config_setting);
  std::shared_ptr<STDescManager> std_manager_prior(new STDescManager(config_setting));

  gtsam::Values initial;
  gtsam::NonlinearFactorGraph graph;

  // https://github.com/TixiaoShan/LIO-SAM/blob/6665aa0a4fcb5a9bb3af7d3923ae4a035b489d47/src/mapOptmization.cpp#L1385
  gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

  gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8)
              .finished()); // rad*rad, meter*meter

  double loopNoiseScore = 1e-1;
  gtsam::Vector robustNoiseVector6(
      6); // gtsam::Pose3 factor has 6 elements (6D)
  robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
      loopNoiseScore, loopNoiseScore, loopNoiseScore;
  gtsam::noiseModel::Base::shared_ptr robustLoopNoise =
      gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Cauchy::Create(1),
          gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  gtsam::ISAM2 isam(parameters);

  size_t cloudInd = 0;
  size_t keyCloudInd = 0;

  std::vector<PointCloud::Ptr> cloud_vec;
  std::vector<Eigen::Affine3d> pose_vec;
  std::vector<Eigen::Affine3d> origin_pose_vec;
  std::vector<Eigen::Affine3d> key_pose_vec; // 回环可视化
  std::vector<Eigen::Affine3d> keyframe_pose_vec;
  std::vector<std::pair<int, int>> loop_container;

  std::vector<Eigen::Affine3d> ref_keyframe_poses;
  std::vector<PointCloud::Ptr> ref_keyframe_clouds;
  // std::vector<std::pair<STDesc, STDesc>> ms_loop_std_pair;
  std::vector<InterSessionLoop> inter_session_loops;            // 累积所有 inter-session loops，用于估计 T_W1_to_W0_est
  Eigen::Affine3d T_W1_to_W0_est = Eigen::Affine3d::Identity(); // ★ 当前估计的会话间变换
  bool has_T_W1_to_W0 = false;                                  // ★ 是否已有有效估计

  PointCloud::Ptr key_cloud(new PointCloud);

  bool has_loop_flag = false;
  gtsam::Values curr_estimate;

  Eigen::Affine3d last_pose;
  last_pose.setIdentity();

  // 启动异步 spinner（放在 while 外）
  ros::AsyncSpinner spinner(2);
  spinner.start();

  // 使用 WallRate，避免 /use_sim_time + ros::Rate 卡死的问题
  ros::WallRate rate(100.0);

  bool keyframes_saved = false; // 标记是否已经保存过关键帧
  ros::WallTime last_data_time = ros::WallTime::now();

  if (config_setting.multi_session_mode_ == 1)
  {
    // 注意这里传的是 session_id = 0，且 std_manager 作为 “参考库”
    if (!loadSessionAndBuildSTD(ref_keyframe_poses,
                                ref_keyframe_clouds,
                                config_setting,
                                0,
                                std_manager_prior))
    {
      ROS_ERROR("Failed to load reference session & build STD DB.");
    }
    else
    {
      ROS_INFO("Reference session loaded, keyframes = %zu",
               ref_keyframe_poses.size());
    }

    // 额外：把 ref_keyframe_clouds + poses 累加一张 ref_map 用于 RViz 显示
    PointCloud::Ptr ref_full_map(new PointCloud);
    sensor_msgs::PointCloud2 map_msg;
    for (auto &cloud : ref_keyframe_clouds)
    {
      pcl::toROSMsg(*cloud, map_msg);
      map_msg.header.frame_id = "camera_init";
      pubRefMap.publish(map_msg);
    }
    // *ref_full_map += *cloud;

    // 发布一个 latched map
  }

  while (ros::ok())
  {
    // ros::spinOose_vence();
    PointCloud::Ptr current_cloud_body(new PointCloud);
    PointCloud::Ptr current_cloud_world(new PointCloud);
    Eigen::Affine3d pose;
    bool flag = syncPackages(current_cloud_body, pose);
    // if (!flag)
    //   cout << flag << endl;
    if (flag)
    {
      last_data_time = ros::WallTime::now();
      auto origin_estimate_affine3d = pose;
      pcl::transformPointCloud(*current_cloud_body, *current_cloud_world, pose);
      down_sampling_voxel(*current_cloud_world, config_setting.ds_size_);
      // down sample body cloud
      down_sampling_voxel(*current_cloud_body, config_setting.ds_size_);
      cloud_vec.push_back(current_cloud_body);
      pose_vec.push_back(pose);
      origin_pose_vec.push_back(pose);
      PointCloud origin_cloud;
      pcl::transformPointCloud(*current_cloud_body, origin_cloud,
                               origin_estimate_affine3d);
      sensor_msgs::PointCloud2 pub_cloud;
      pcl::toROSMsg(origin_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubOriginCloud.publish(pub_cloud);

      Eigen::Quaterniond _r(origin_estimate_affine3d.rotation());
      nav_msgs::Odometry odom;
      odom.header.frame_id = "camera_init";
      odom.pose.pose.position.x = origin_estimate_affine3d.translation().x();
      odom.pose.pose.position.y = origin_estimate_affine3d.translation().y();
      odom.pose.pose.position.z = origin_estimate_affine3d.translation().z();
      odom.pose.pose.orientation.w = _r.w();
      odom.pose.pose.orientation.x = _r.x();
      odom.pose.pose.orientation.y = _r.y();
      odom.pose.pose.orientation.z = _r.z();
      pubOdomOrigin.publish(odom);

      *key_cloud += *current_cloud_world;
      initial.insert(cloudInd, gtsam::Pose3(pose.matrix()));

      if (!cloudInd)
      {
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(
            0, gtsam::Pose3(pose.matrix()), odometryNoise));
        // keyframe_pose_vec.push_back(pose);
      }
      else
      {
        auto prev_pose = gtsam::Pose3(origin_pose_vec[cloudInd - 1].matrix());
        auto curr_pose = gtsam::Pose3(pose.matrix());
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            cloudInd - 1, cloudInd, prev_pose.between(curr_pose),
            odometryNoise));
      }

      // check if keyframe
      if (cloudInd % config_setting.sub_frame_num_ == 0 && cloudInd != 0)
      {
        ROS_INFO("key frame idx: [%d], key cloud size: [%d]", (int)keyCloudInd,
                 (int)key_cloud->size());
        // step1. Descriptor Extraction
        std::vector<STDesc> stds_vec;
        std_manager->GenerateSTDescs(key_cloud, stds_vec);

        // step2. Searching Loop
        std::pair<int, double> search_result(-1, 0);
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        loop_transform.first << 0, 0, 0;
        loop_transform.second = Eigen::Matrix3d::Identity();
        std::vector<std::pair<STDesc, STDesc>> loop_std_pair;

        if (keyCloudInd > config_setting.skip_near_num_)
        {
          std_manager->SearchLoop(stds_vec, search_result, loop_transform,
                                  loop_std_pair, std_manager->data_base_);
        }

        // step3. Add descriptors to the database
        std_manager->AddSTDescs(stds_vec);

        // ----------------- multi-session loop detection（session1 → session0） -----------------
        std::pair<int, double> ms_search_result(-1, 0);
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> ms_loop_transform;
        ms_loop_transform.first << 0, 0, 0;
        ms_loop_transform.second = Eigen::Matrix3d::Identity();
        std::vector<std::pair<STDesc, STDesc>> ms_loop_std_pair;
        if (config_setting.multi_session_mode_ == 1)
        {
          // 1) 用 prior manager 再生成一遍当前 keyframe 的 ST 描述子（保证平面索引正确）
          std::vector<STDesc> ms_stds_vec;
          std_manager_prior->GenerateSTDescs(key_cloud, ms_stds_vec);

          if (!ms_stds_vec.empty())
          {
            // 2) 在 session0 的 STD 数据库上搜索 inter-session loop
            std_manager_prior->SearchLoop(
                ms_stds_vec,
                ms_search_result,
                ms_loop_transform,
                ms_loop_std_pair,
                std_manager_prior->data_base_,
                /*mode =*/1);

            int ms_match_kf = ms_search_result.first;  // session0 中的 keyframe id
            double ms_score = ms_search_result.second; // 匹配得分

            if (ms_match_kf >= 0 && ms_score > config_setting.inter_session_icp_threshold_)
            {
              // 3) 平面几何 ICP 再 refinement（几何一致性验证）
              std_manager_prior->PlaneGeometricIcp(
                  std_manager_prior->plane_cloud_vec_.back(),       // 当前 keyframe 的平面（session1）
                  std_manager_prior->plane_cloud_vec_[ms_match_kf], // session0 中匹配的平面
                  ms_loop_transform);

              // 4) 把当前这次 inter-session loop 记录到数组里，后面用 estimateSessionTransform 做加权平均
              InterSessionLoop ms_loop;
              ms_loop.ref_kf = ms_match_kf;                            // session0 keyframe id
              ms_loop.cur_kf = keyCloudInd;                            // session1 keyframe id（当前 keyframe 的索引）
              ms_loop.score = ms_score;                                // 用 icp / matching score 当权重
              ms_loop.relative_pose.first = ms_loop_transform.first;   // t
              ms_loop.relative_pose.second = ms_loop_transform.second; // R
              ms_loop.match_pairs = ms_loop_std_pair;                  // 可选，调试可用

              inter_session_loops.push_back(ms_loop);

              // 5) 使用所有累计的 inter-session loops，估计加权平均的 T_W1_to_W0_est
              T_W1_to_W0_est = estimateSessionTransform(inter_session_loops, true);
              has_T_W1_to_W0 = true;
              // 日志：输出累计使用了多少个 loop，以及当前估计的 T_W1_to_W0_est
              Eigen::Vector3d t_w1w0 = T_W1_to_W0_est.translation();
              Eigen::Vector3d rpy = T_W1_to_W0_est.rotation().eulerAngles(2, 1, 0); // yaw-pitch-roll

              ROS_INFO_STREAM("[MS Loop] use " << inter_session_loops.size()
                                               << " loops, new loop: session1 kf " << keyCloudInd
                                               << ", session0 kf " << ms_match_kf
                                               << ", score = " << ms_score
                                               << ", T_W1_to_W0_est translation = "
                                               << t_w1w0.transpose()
                                               << ", yaw(deg) = " << rpy[0] * 57.3);

              // 6) 可视化：画一条红色线把当前 keyframe 和匹配的 prior keyframe 连起来
              //    这里的可视化仍然用当前估计的位姿，不依赖 T_W1_to_W0_est
              // Eigen::Affine3d pose_cur_w1 = keyframe_pose_vec.back();        // 当前 session1 keyframe 在 W1 下
              // Eigen::Affine3d pose_ref_w0 = ref_keyframe_poses[ms_match_kf]; // session0 keyframe 在 W0 下

              // visualizeMultiSessionLoop(pubMSLoopConstraintEdge,
              //                           pose_cur_w1,
              //                           pose_ref_w0);
            }
          }
        }

        // publish
        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(*key_cloud, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCurrentCloud.publish(pub_cloud);
        pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCurrentCorner.publish(pub_cloud);

        std_manager->key_cloud_vec_.push_back(key_cloud->makeShared());
        keyframe_pose_vec.push_back(pose);
        if (search_result.first > 0)
        {
          std::cout << "[Loop Detection] triggle loop: " << keyCloudInd << "--"
                    << search_result.first << ", score:" << search_result.second
                    << std::endl;

          has_loop_flag = true;
          int match_frame = search_result.first;
          // obtain optimal transform
          std_manager->PlaneGeometricIcp(
              std_manager->plane_cloud_vec_.back(),
              std_manager->plane_cloud_vec_[match_frame], loop_transform);

          // std::cout << "delta transform:" << std::endl;
          // std::cout << "translation: " << loop_transform.first.transpose() <<
          // std::endl;

          // auto euler = loop_transform.second.eulerAngles(2, 1, 0) * 57.3;
          // std::cout << "rotation(ypr): " << euler[0] << ' ' << euler[1] << '
          // ' << euler[2]
          //           << std::endl;

          /*
            add connection between loop frame.
            e.g. if src_key_frame_id 5 with sub frames 51~60 triggle loop with
                  tar_key_frame_id 1 with sub frames 11~20, add connection
            between each sub frame, 51-11, 52-12,...,60-20.

          */
          int sub_frame_num = config_setting.sub_frame_num_;
          for (size_t j = 1; j <= sub_frame_num; j++)
          {
            int src_frame = cloudInd + j - sub_frame_num;

            auto delta_T = Eigen::Affine3d::Identity();
            delta_T.translate(loop_transform.first);
            delta_T.rotate(loop_transform.second);
            Eigen::Affine3d src_pose_refined = delta_T * pose_vec[src_frame];

            int tar_frame = match_frame * sub_frame_num + j;
            // old
            // Eigen::Affine3d tar_pose = pose_vec[tar_frame];
            Eigen::Affine3d tar_pose = origin_pose_vec[tar_frame];

            loop_container.push_back({tar_frame, src_frame});

            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                tar_frame, src_frame,
                gtsam::Pose3(tar_pose.matrix())
                    .between(gtsam::Pose3(src_pose_refined.matrix())),
                robustLoopNoise));
          }

          pcl::toROSMsg(*std_manager->key_cloud_vec_[search_result.first],
                        pub_cloud);
          pub_cloud.header.frame_id = "camera_init";
          pubMatchedCloud.publish(pub_cloud);

          pcl::toROSMsg(*std_manager->corner_cloud_vec_[search_result.first],
                        pub_cloud);
          pub_cloud.header.frame_id = "camera_init";
          pubMatchedCorner.publish(pub_cloud);
          publish_std_pairs(loop_std_pair, pubSTD);
        }

        key_cloud->clear();
        ++keyCloudInd;
      }
      isam.update(graph, initial);
      isam.update();

      if (has_loop_flag)
      {
        isam.update();
        isam.update();
        isam.update();
        isam.update();
        isam.update();
      }

      graph.resize(0);
      initial.clear();

      curr_estimate = isam.calculateEstimate();
      update_poses(curr_estimate, pose_vec);

      if (has_loop_flag)
      {
        // publish correct cloud map
        PointCloud full_map;
        for (int i = 0; i < pose_vec.size(); ++i)
        {
          PointCloud correct_cloud;
          pcl::transformPointCloud(*cloud_vec[i], correct_cloud, T_W1_to_W0_est * pose_vec[i]);
          full_map += correct_cloud;
        }
        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(full_map, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCorrectCloud.publish(pub_cloud);

        // publish corerct path
        nav_msgs::Path correct_path;
        // for (int i = 0; i < pose_vec.size(); i += 1)
        // {

        //   geometry_msgs::PoseStamped msg_pose;
        //   msg_pose.pose.position.x = pose_vec[i].translation()[0];
        //   msg_pose.pose.position.y = pose_vec[i].translation()[1];
        //   msg_pose.pose.position.z = pose_vec[i].translation()[2];
        //   Eigen::Quaterniond pose_q(pose_vec[i].rotation());
        //   msg_pose.header.frame_id = "camera_init";
        //   msg_pose.pose.orientation.x = pose_q.x();
        //   msg_pose.pose.orientation.y = pose_q.y();
        //   msg_pose.pose.orientation.z = pose_q.z();
        //   msg_pose.pose.orientation.w = pose_q.w();
        //   correct_path.poses.push_back(msg_pose);
        // }
        correct_path.header.stamp = ros::Time::now();
        correct_path.header.frame_id = "camera_init";
        for (int i = 0; i < pose_vec.size(); i += 1)
        {
          Eigen::Affine3d pose_w0 = T_W1_to_W0_est * pose_vec[i];

          geometry_msgs::PoseStamped msg_pose;
          msg_pose.pose.position.x = pose_w0.translation()[0];
          msg_pose.pose.position.y = pose_w0.translation()[1];
          msg_pose.pose.position.z = pose_w0.translation()[2];
          Eigen::Quaterniond pose_q(pose_w0.rotation());
          msg_pose.header.frame_id = "camera_init";
          msg_pose.pose.orientation.x = pose_q.x();
          msg_pose.pose.orientation.y = pose_q.y();
          msg_pose.pose.orientation.z = pose_q.z();
          msg_pose.pose.orientation.w = pose_q.w();
          correct_path.poses.push_back(msg_pose);
        }
        pubCorrectPath.publish(correct_path);
      }

      visualizeLoopClosure(pubLoopConstraintEdge, loop_container, pose_vec);

      has_loop_flag = false;
      ++cloudInd;
    }
    else
    {
      // ROS_INFO_THROTTLE(1.0, "not syncing"); // 1 秒打印一次，避免刷屏
      ros::WallDuration no_data_duration =
          ros::WallTime::now() - last_data_time;

      if (!keyframes_saved &&
          !keyframe_pose_vec.empty() &&
          config_setting.keyframe_save_ &&
          no_data_duration.toSec() > 5.0)
      {
        keyframes_saved = true;
        ROS_INFO("saving keyframe ...");
        boost::filesystem::create_directories(config_setting.pos_dir_);
        boost::filesystem::create_directories(config_setting.std_dir_);
        boost::filesystem::create_directories(config_setting.pcd_dir_);
        string pose_file_name = config_setting.pos_dir_ + "poses.txt";
        string std_file_name = config_setting.std_dir_ + "std_database.txt";
        std_manager->saveDatabase(std_file_name);
        std::ofstream pose_file(pose_file_name);

        for (int i = 0; i < keyframe_pose_vec.size(); ++i)
        {
          std::ostringstream oss;
          oss << std::setw(4) << std::setfill('0') << i;
          string key_frame_idx = oss.str();
          PointCloud correct_cloud = *std_manager->key_cloud_vec_[i];
          down_sampling_voxel(correct_cloud, 0.05);
          pcl::io::savePCDFileBinary(config_setting.pcd_dir_ + key_frame_idx + ".pcd", correct_cloud);
          Eigen::Quaterniond q(keyframe_pose_vec[i].rotation());
          q.normalize();

          pose_file << keyframe_pose_vec[i].translation()[0] << ", "
                    << keyframe_pose_vec[i].translation()[1] << ","
                    << keyframe_pose_vec[i].translation()[2] << ","
                    << q.w() << ","
                    << q.x() << ","
                    << q.y() << ","
                    << q.z() << "\n";
        }
        pose_file.close();
        ROS_INFO("saving done!");

        // -- -- -- -- --新增：保存“对齐到 session0 世界系 W0 的全局地图”-- -- -- -- --
        if (has_T_W1_to_W0)
        {
          ROS_INFO("Saving aligned global map (session1 warped to session0 frame W0)...");
          ROS_INFO_STREAM("Keyframe size of prior session:"
                          << ref_keyframe_clouds.size()
                          << "\nKeyframe size of current session:"
                          << std_manager->key_cloud_vec_.size());
          // 1) 把当前 session1 的关键帧点云从 W1 映射到 W0
          PointCloud aligned_session1_map;
          for (size_t i = 0; i < std_manager->key_cloud_vec_.size(); ++i)
          {
            PointCloud transformed;
            // key_cloud_vec_[i] 已经在 W1 下，直接乘 T_W1_to_W0_est 即可变到 W0
            pcl::transformPointCloud(*std_manager->key_cloud_vec_[i],
                                     transformed,
                                     T_W1_to_W0_est);
            aligned_session1_map += transformed;
          }
          ROS_INFO_STREAM("aligned_session1_map size = " << aligned_session1_map.size());

          // 2) 把先验 session0 的 keyframe 点云（本来就在 W0）叠加上去，得到最终全局地图
          PointCloud final_global_map = aligned_session1_map;
          for (const auto &cloud_ref : ref_keyframe_clouds)
          {
            final_global_map += *cloud_ref;
          }
          ROS_INFO_STREAM("final_global_map size before downsample = "
                          << final_global_map.size());

          // 3) 下采样再保存
          ROS_INFO("Start down_sampling_voxel...");
          down_sampling_voxel(final_global_map, 0.05);
          ROS_INFO("Finish down_sampling_voxel.");

          std::string aligned_dir = config_setting.pos_dir_ + "aligned/";
          boost::filesystem::create_directories(aligned_dir);
          std::string aligned_map_file = aligned_dir + "global_map_W0.pcd";

          ROS_INFO_STREAM("Start savePCDFileBinary: " << aligned_map_file);
          int ret = pcl::io::savePCDFileBinary(aligned_map_file, final_global_map);
          ROS_INFO_STREAM("Aligned global map saved to: " << aligned_map_file);
          if (ret < 0)
          {
            ROS_ERROR_STREAM("savePCDFileBinary failed, code = " << ret
                                                                 << ", path = " << aligned_map_file);
          }
          else
          {
            ROS_INFO_STREAM("Aligned global map saved to: " << aligned_map_file
                                                            << ", points = " << final_global_map.size());
          }

          ROS_INFO_STREAM("Aligned global map saved to: " << aligned_map_file);
        }
        else
        {
          ROS_WARN("No valid T_W1_to_W0 estimation yet, skip saving aligned global map.");
        }
      }
    }
  }
  return 0;
}
