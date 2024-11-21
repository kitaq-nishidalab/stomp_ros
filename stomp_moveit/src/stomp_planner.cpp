/**
 * @file stomp_planner.cpp
 * @brief This defines the stomp planner for MoveIt
 *
 * @author Jorge Nicho
 * @date April 4, 2016
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2016, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ros/ros.h>
#include <moveit/robot_state/conversions.h>
#include <stomp_moveit/stomp_planner.h>
#include <class_loader/class_loader.hpp>
#include <stomp/utils.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <stomp_moveit/utils/kinematics.h>
#include <stomp_moveit/utils/polynomial.h>
#include <stomp_moveit/PathSeed.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <sstream>
#include <string>
#include <stomp_moveit/noise_generators/normal_distribution_sampling.h>



static const std::string DEBUG_NS = "stomp_planner";
static const std::string DESCRIPTION = "STOMP";
static const double TIMEOUT_INTERVAL = 0.5;
static int const IK_ATTEMPTS = 4;
static int const IK_TIMEOUT = 0.005;
const static double MAX_START_DISTANCE_THRESH = 0.5;

/**
 * @brief Parses a XmlRpcValue and populates a StompComfiguration structure.
 * @param config        The XmlRpcValue of stomp configuration parameters
 * @param group         The moveit planning group
 * @param stomp_config  The stomp configuration structure
 * @return True if successfully parsed, otherwise false.
 */
bool parseConfig(XmlRpc::XmlRpcValue config,const moveit::core::JointModelGroup* group,stomp::StompConfiguration& stomp_config)
{
  using namespace XmlRpc;
  // Set default values for optional config parameters
  stomp_config.control_cost_weight = 0.0;
  stomp_config.initialization_method = 1; // LINEAR_INTERPOLATION
  stomp_config.num_timesteps = 40;
  stomp_config.delta_t = 1.0;
  stomp_config.num_iterations = 50;
  stomp_config.num_iterations_after_valid = 0;
  stomp_config.max_rollouts = 100;
  stomp_config.num_rollouts = 10;
  stomp_config.exponentiated_cost_sensitivity = 10.0;

  ROS_INFO("-------------------- STOMP Configuration --------------------");
  ROS_INFO("Default values set: control_cost_weight: %f, initialization_method: %d, num_timesteps: %d, delta_t: %f, num_iterations: %d, num_iterations_after_valid: %d, max_rollouts: %d, num_rollouts: %d, exponentiated_cost_sensitivity: %f",
           stomp_config.control_cost_weight, stomp_config.initialization_method, stomp_config.num_timesteps, stomp_config.delta_t, stomp_config.num_iterations, stomp_config.num_iterations_after_valid, stomp_config.max_rollouts, stomp_config.num_rollouts, stomp_config.exponentiated_cost_sensitivity);

  // Load optional config parameters if they exist
  if (config.hasMember("control_cost_weight"))
    stomp_config.control_cost_weight = static_cast<double>(config["control_cost_weight"]);
    // ROS_INFO("control_cost_weight: %f", stomp_config.control_cost_weight);

  if (config.hasMember("initialization_method"))
    stomp_config.initialization_method = static_cast<int>(config["initialization_method"]);
    // ROS_INFO("initialization_method: %d", stomp_config.initialization_method);

  if (config.hasMember("num_timesteps"))
    stomp_config.num_timesteps = static_cast<int>(config["num_timesteps"]);
    // ROS_INFO("num_timesteps: %d", stomp_config.num_timesteps);

  if (config.hasMember("delta_t"))
    stomp_config.delta_t = static_cast<double>(config["delta_t"]);
    // ROS_INFO("delta_t: %f", stomp_config.delta_t);

  if (config.hasMember("num_iterations"))
    stomp_config.num_iterations = static_cast<int>(config["num_iterations"]);
    // ROS_INFO("num_iterations: %d", stomp_config.num_iterations);

  if (config.hasMember("num_iterations_after_valid"))
    stomp_config.num_iterations_after_valid = static_cast<int>(config["num_iterations_after_valid"]);
    // ROS_INFO("num_iterations_after_valid: %d", stomp_config.num_iterations_after_valid);

  if (config.hasMember("max_rollouts"))
    stomp_config.max_rollouts = static_cast<int>(config["max_rollouts"]);
    // ROS_INFO("max_rollouts: %d", stomp_config.max_rollouts);

  if (config.hasMember("num_rollouts"))
    stomp_config.num_rollouts = static_cast<int>(config["num_rollouts"]);
    // ROS_INFO("num_rollouts: %d", stomp_config.num_rollouts);

  if (config.hasMember("exponentiated_cost_sensitivity"))
    stomp_config.exponentiated_cost_sensitivity = static_cast<int>(config["exponentiated_cost_sensitivity"]);
    // ROS_INFO("exponentiated_cost_sensitivity: %f", stomp_config.exponentiated_cost_sensitivity);

  // getting number of joints
  stomp_config.num_dimensions = group->getActiveJointModels().size();
  if(stomp_config.num_dimensions == 0)
  {
    ROS_ERROR("Planning Group %s has no active joints",group->getName().c_str());
    return false;
  }
  return true;
}

namespace stomp_moveit
{

StompPlanner::StompPlanner(const std::string& group,const XmlRpc::XmlRpcValue& config,
                           const moveit::core::RobotModelConstPtr& model):
    PlanningContext(DESCRIPTION,group),
    config_(config),
    robot_model_(model),
    ik_solver_(new utils::kinematics::IKSolver(model,group)),
    ph_(new ros::NodeHandle("~")),
    rows_(0),
    cols_(0),
    parameters_(Eigen::MatrixXd::Zero(1,1))
{
  // PathSeed subscriber
  path_seed_subscriber_ = ph_->subscribe("path_seed", 500, &StompPlanner::pathSeedCallback, this);
  setup();
}

// callbaxck function for the path_seed topic
void StompPlanner::pathSeedCallback(const stomp_moveit::PathSeed::ConstPtr& msg)
{
    rows_ = msg->rows;
    cols_ = msg->cols;

    parameters_ = Eigen::MatrixXd(rows_, cols_);
    int index = 0;
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            parameters_(i, j) = msg->data[index++];
        }
    }
    
    // For debugging
    // ROS_INFO("Received path seed:");
    // ROS_INFO("  Rows: %d", rows_);
    // ROS_INFO("  Cols: %d", cols_);
    // // Format the matrix using IOFormat
    // Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
    // std::stringstream ss;
    // ss << parameters_.format(fmt);
    // // Convert to std::string
    // std::string parameters_str = ss.str();
    // ROS_INFO_STREAM("Received path seed:\n" << parameters_str);
}

StompPlanner::~StompPlanner()
{
}

void StompPlanner::setParameters(const Eigen::MatrixXd& parameters, int rows, int cols) {
    // For debugging
    // // Format the matrix using IOFormat
    // Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
    // std::stringstream ss;
    // ss << parameters.format(fmt);
    // // Convert to std::string
    // std::string parameters_str = ss.str();
    // ROS_INFO("Setting parameters, rows: %d, cols: %d, parameters: %d", rows_, cols_, parameters_str);
    parameters_ = parameters;
    rows_ = rows;
    cols_ = cols;
}

Eigen::MatrixXd StompPlanner::getInitialParameters() const {
    // For debugging
    // ROS_INFO("Getting initial parameters:");
    // // Format the matrix using IOFormat
    // Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
    // std::stringstream ss;
    // ss << parameters_.format(fmt);
    // // Convert to std::string
    // std::string parameters_str = ss.str();
    // ROS_INFO_STREAM("Initial parameters:\n" << parameters_str);

    return parameters_;
}

int StompPlanner::getRows() const {
    // For debugging
    // ROS_INFO("Getting rows:");
    // ROS_INFO("  Rows: %d", rows_);
    return rows_;
}

int StompPlanner::getCols() const {
    // For debugging
    // ROS_INFO("Getting cols:");
    // ROS_INFO("  Cols: %d", cols_);
    return cols_;
}

void StompPlanner::setup()
{
  if(!getPlanningScene())
  {
    setPlanningScene(planning_scene::PlanningSceneConstPtr(new planning_scene::PlanningScene(robot_model_)));
  }

  // loading parameters
  try
  {
    // creating tasks
    XmlRpc::XmlRpcValue task_config;
    task_config = config_["task"];
    // ログ出力
    // ROS_INFO("Task Config: %s", task_config.toXml().c_str());  // XML形式でログ出力
    task_.reset(new StompOptimizationTask(robot_model_,group_,task_config));

    if(!robot_model_->hasJointModelGroup(group_))
    {
      std::string msg = "Stomp Planning Group '" + group_ + "' was not found";
      ROS_ERROR("%s",msg.c_str());
      throw std::logic_error(msg);
    }

    // parsing stomp parameters
    if(!config_.hasMember("optimization") || !parseConfig(config_["optimization" ],robot_model_->getJointModelGroup(group_),stomp_config_))
    {
      std::string msg = "Stomp 'optimization' parameter for group '" + group_ + "' failed to load";
      ROS_ERROR("%s", msg.c_str());
      throw std::logic_error(msg);
    }

    stomp_.reset(new stomp::Stomp(stomp_config_,task_));
  }
  catch(XmlRpc::XmlRpcException& e)
  {
    throw std::logic_error("Stomp Planner failed to load configuration for group '" + group_+"'; " + e.getMessage());
  }

}


// モーションプランニングを実行し、プランニングが成功したかどうかを返す
bool StompPlanner::solve(planning_interface::MotionPlanResponse &res)
{
  // プランニングの開始時刻を記録
  ros::WallTime start_time = ros::WallTime::now();
  
  // 詳細なプランニングレスポンス用オブジェクト
  planning_interface::MotionPlanDetailedResponse detailed_res;

  // solve(detailed_res) によってモーションプランニングを実行し、成功したかどうかを success に格納
  bool success = solve(detailed_res);

  // プランニングが成功した場合、最終的な trajectory をレスポンスに設定
  if (success)
  {
    res.trajectory_ = detailed_res.trajectory_.back();  // trajectory の最後の軌道を res に格納
  }

  // プランニングにかかった時間を計算し、レスポンスに設定
  ros::WallDuration wd = ros::WallTime::now() - start_time;
  res.planning_time_ = ros::Duration(wd.sec, wd.nsec).toSec();

  // エラーコードをレスポンスに設定
  res.error_code_ = detailed_res.error_code_;

  // プランニングの成功/失敗を返す
  return success;
}


bool StompPlanner::solve(planning_interface::MotionPlanDetailedResponse &res)
{
  bool use_pathseed = false;

  // ROSパラメータから値を取得
  if (!ros::param::get("use_pathseed", use_pathseed))
  {
      // パラメータが設定されていない場合はデフォルト値を使用
      use_pathseed = false;
      ROS_WARN("Parameter 'use_pathseed' not set. Using default value: false");
  }
  ROS_INFO("use_pathseed: %s", use_pathseed ? "true" : "false");

  using namespace stomp;

  // initializing response
  res.description_.resize(1,"plan");
  res.processing_time_.resize(1);
  res.trajectory_.resize(1);
  res.error_code_.val = moveit_msgs::MoveItErrorCodes::SUCCESS;

  ros::WallTime start_time = ros::WallTime::now();
  bool success = false;

  trajectory_msgs::JointTrajectory trajectory;
  Eigen::MatrixXd parameters;
  bool planning_success;

  // Load optimization parameters from rosparam
  ros::param::get("move_group/stomp/xarm6/optimization/num_timesteps", stomp_config_.num_timesteps);
  ros::param::get("move_group/stomp/xarm6/optimization/num_iterations", stomp_config_.num_iterations);
  ros::param::get("move_group/stomp/xarm6/optimization/num_iterations_after_valid", stomp_config_.num_iterations_after_valid);
  ros::param::get("move_group/stomp/xarm6/optimization/num_rollouts", stomp_config_.num_rollouts);
  ros::param::get("move_group/stomp/xarm6/optimization/max_rollouts", stomp_config_.max_rollouts);
  ros::param::get("move_group/stomp/xarm6/optimization/initialization_method", stomp_config_.initialization_method);
  ros::param::get("move_group/stomp/xarm6/optimization/control_cost_weight", stomp_config_.control_cost_weight);

  // Load task parameters from rosparam
  // Load parameters from rosparam
  // try {
  //     // Load noise generator parameters
  //     XmlRpc::XmlRpcValue noise_generator;
  //     if (ros::param::get("move_group/stomp/xarm6/task/noise_generator", noise_generator)) {
  //         if (noise_generator.getType() == XmlRpc::XmlRpcValue::TypeArray && noise_generator.size() > 0) {
  //             if (noise_generator[0].hasMember("stddev") && 
  //                 noise_generator[0]["stddev"].getType() == XmlRpc::XmlRpcValue::TypeArray) {
  //                 ROS_INFO("stddev values:");
  //                 for (int i = 0; i < noise_generator[0]["stddev"].size(); ++i) {
  //                     ROS_INFO("stddev[%d]: %f", i, static_cast<double>(noise_generator[0]["stddev"][i]));
  //                 }
  //             }
  //         }
  //     } else {
  //         ROS_ERROR("Failed to get noise_generator parameters.");
  //     }

      // TODO: taskパラメータに関してはparamを読み込ませてもMoveItには反映されない問題
      // Load cost function parameters
      // XmlRpc::XmlRpcValue cost_functions;
      // if (ros::param::get("move_group/stomp/xarm6/task/cost_functions", cost_functions)) {
      //     if (cost_functions.getType() == XmlRpc::XmlRpcValue::TypeArray) {
      //         // ROS_INFO("Cost function parameters: %s", cost_functions.toXml().c_str());

      //         // 配列の最初の要素にアクセス
      //         XmlRpc::XmlRpcValue first_function = cost_functions[0];

      //         // メンバーの存在を確認し出力
      //         if (first_function.hasMember("class")) {
      //             ROS_INFO("Cost function class: %s", std::string(first_function["class"]).c_str());
      //         }
      //         if (first_function.hasMember("collision_penalty")) {
      //             ROS_INFO("Collision penalty: %f", static_cast<double>(first_function["collision_penalty"]));
      //         }
      //         if (first_function.hasMember("cost_weight")) {
      //             ROS_INFO("Cost weight: %f", static_cast<double>(first_function["cost_weight"]));
      //         }
      //         if (first_function.hasMember("kernel_window_percentage")) {
      //             ROS_INFO("Kernel window percentage: %f", static_cast<double>(first_function["kernel_window_percentage"]));
      //         }
      //         if (first_function.hasMember("longest_valid_joint_move")) {
      //             ROS_INFO("Longest valid joint move: %f", static_cast<double>(first_function["longest_valid_joint_move"]));
      //         }
      //     } else {
      //         ROS_WARN("cost_functions is not an array.");
      //     }
      // } else {
      //     ROS_ERROR("Failed to get cost_functions parameters.");
      // }
  // } catch (const std::exception& e) {
  //     ROS_ERROR("Exception occurred: %s", e.what());
  // }


  // debug
  // ROS_INFO("num_timesteps: %d", stomp_config_.num_timesteps);
  // ROS_INFO("num_iterations: %d", stomp_config_.num_iterations);
  // ROS_INFO("num_iterations_after_valid: %d", stomp_config_.num_iterations_after_valid);
  // ROS_INFO("num_rollouts: %d", stomp_config_.num_rollouts);
  // ROS_INFO("max_rollouts: %d", stomp_config_.max_rollouts);
  // ROS_INFO("initialization_method: %d", stomp_config_.initialization_method);
  // ROS_INFO("control_cost_weight: %f", stomp_config_.control_cost_weight);

  // look for seed trajectory
  Eigen::MatrixXd initial_parameters;
  bool use_seed = getSeedParameters(initial_parameters);
  ROS_INFO("use_seed: %s", use_seed ? "true" : "false");

  // create timeout timer
  ros::WallDuration allowed_time(request_.allowed_planning_time);
  ROS_WARN_COND(TIMEOUT_INTERVAL > request_.allowed_planning_time,
                "%s allowed planning time %f is less than the minimum planning time value of %f",
                getName().c_str(),request_.allowed_planning_time,TIMEOUT_INTERVAL);
  std::atomic<bool> terminating(false);
  ros::Timer timeout_timer = ph_->createTimer(ros::Duration(TIMEOUT_INTERVAL), [&](const ros::TimerEvent& evnt)
  {
    if(((ros::WallTime::now() - start_time) > allowed_time))
    {
      ROS_ERROR_COND(!terminating,"%s exceeded allowed time of %f , terminating",getName().c_str(),allowed_time.toSec());
      this->terminate();
      terminating = true;
    }

  },false);

  if (use_seed)
  {
    ROS_INFO("%s Seeding trajectory from MotionPlanRequest",getName().c_str());

    // updating time step in stomp configuraion
    stomp_config_.num_timesteps = initial_parameters.cols();
    
    // setting up up optimization task
    if(!task_->setMotionPlanRequest(planning_scene_, request_, stomp_config_, res.error_code_))
    {
      res.error_code_.val = moveit_msgs::MoveItErrorCodes::FAILURE;
      return false;
    }

    stomp_->setConfig(stomp_config_);
    planning_success = stomp_->solve(initial_parameters, parameters);
  }

  else {
    // declearing the planner
    StompPlanner planner("xarm6", config_, robot_model_);

    // Check and obtain initial parameters, retry if necessary
    const int max_retries = 5;  // Maximum number of retries
    const double retry_frequency = 50.0;  // Retry frequency in Hz

    Eigen::MatrixXd initial_parameters;
    int rows, cols;
    int retries = 0;

    ROS_INFO("use_pathseed: %s", use_pathseed ? "true" : "false");
    if (use_pathseed == true) {  
      ros::Rate rate(retry_frequency);  // Rate object to control the loop frequencys
      // Loop to retry obtaining initial_parameters if they are empty
      while (retries < max_retries) {
          ROS_INFO("Attempting to obtain initial parameters from the PathSeed...");
          initial_parameters = planner.getInitialParameters();
          rows = planner.getRows();
          cols = planner.getCols();
          std::cout << "rows: " << rows << ", cols: " << cols << std::endl;
          // Check if the initial_parameters are valid (non-zero size)
          if (rows > 0 && cols > 0) {
              ROS_INFO("Initial parameters obtained successfully.");
              break;
          } else {
              ROS_WARN("Initial parameters are empty. Retrying...");
              retries++;
              if (retries == max_retries) {
                  ROS_WARN("Failed to obtain initial parameters after %d retries.", max_retries);
                  use_pathseed = false;  // Set use_pathseed to false to proceed without pathseed    
              }
          }
          rate.sleep();  // Wait until the next cycle
      }
      // If we are using pathseed, proceed with pathseed-based planning
      ROS_INFO("|--------------------------------------------------|");
      ROS_INFO("|  Initial Parameters from the PathSeed you set    |");
      ROS_INFO("|--------------------------------------------------|");

      // ROS_WARN("----------------- Get Parameters -----------------");
      // Load optimization parameters from rosparam
      ros::param::get("move_group/stomp/xarm6/optimization/num_timesteps", stomp_config_.num_timesteps);
      ros::param::get("move_group/stomp/xarm6/optimization/num_iterations", stomp_config_.num_iterations);
      ros::param::get("move_group/stomp/xarm6/optimization/num_iterations_after_valid", stomp_config_.num_iterations_after_valid);
      ros::param::get("move_group/stomp/xarm6/optimization/num_rollouts", stomp_config_.num_rollouts);
      ros::param::get("move_group/stomp/xarm6/optimization/max_rollouts", stomp_config_.max_rollouts);
      ros::param::get("move_group/stomp/xarm6/optimization/initialization_method", stomp_config_.initialization_method);
      ros::param::get("move_group/stomp/xarm6/optimization/control_cost_weight", stomp_config_.control_cost_weight);

      // debug
      // ROS_INFO("num_timesteps: %d", stomp_config_.num_timesteps);
      // ROS_INFO("num_iterations: %d", stomp_config_.num_iterations);
      // ROS_INFO("num_iterations_after_valid: %d", stomp_config_.num_iterations_after_valid);
      // ROS_INFO("num_rollouts: %d", stomp_config_.num_rollouts);
      // ROS_INFO("max_rollouts: %d", stomp_config_.max_rollouts);
      // ROS_INFO("initialization_method: %d", stomp_config_.initialization_method);
      // ROS_INFO("control_cost_weight: %f", stomp_config_.control_cost_weight);

      // For debugging the initial parameters
      // Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
      // std::stringstream ss;
      // ss << initial_parameters.format(fmt);
      // std::string parameters_str = ss.str();

      // std::cout << "rows: " << rows << ", cols: " << cols << std::endl;
      // ROS_INFO_STREAM("Initial parameters:\n" << parameters_str);

      Eigen::MatrixXd initial_parameters_transpose = initial_parameters.transpose();
      stomp_config_.num_timesteps = initial_parameters_transpose.cols();

      // Get the start and goal positions
      Eigen::VectorXd start, goal;
      if (!getStartAndGoal(start, goal)) {
          res.error_code_.val = moveit_msgs::MoveItErrorCodes::INVALID_MOTION_PLAN;
          return false;
      }

      // Ensure the initial trajectory starts and ends at the correct positions
      if (start.size() == initial_parameters_transpose.rows() && 
          goal.size() == initial_parameters_transpose.rows()) {
          initial_parameters_transpose.col(0) = start;  // Set start
          initial_parameters_transpose.col(initial_parameters_transpose.cols() - 1) = goal;  // Set goal
          ROS_INFO("Updated the initial trajectory to match the start and goal.");
      } else {
          ROS_ERROR("Mismatch between the number of joints in start/goal and the pathseed.");
          res.error_code_.val = moveit_msgs::MoveItErrorCodes::FAILURE;
          return false;
      }

      // ROS_WARN("----------------- Settting up Optimization Task -----------------");
      // Setting up optimization task
      if (!task_->setMotionPlanRequest(planning_scene_, request_, stomp_config_, res.error_code_)) {
          ROS_ERROR("Failed to set motion plan request. Check your PathSeed and request parameters.");
          res.error_code_.val = moveit_msgs::MoveItErrorCodes::FAILURE;
          return false;
      }
      stomp_->setConfig(stomp_config_);

      // ROS_WARN("----------------- Solve -----------------");
      planning_success = stomp_->solve(initial_parameters_transpose, parameters);
      ROS_INFO("Planning success: %s", planning_success ? "true" : "false");
      ROS_INFO("----------------- End of PathSeed -----------------");
    }
    else {
      // if you don't want to use the pathseeds you set
      ROS_INFO("Not using pathseed you set.");
      // extracting start and goal
      Eigen::VectorXd start, goal;
      if(!getStartAndGoal(start,goal))
      {
        res.error_code_.val = moveit_msgs::MoveItErrorCodes::INVALID_MOTION_PLAN;
        return false;
      }

      // setting up up optimization task
      if(!task_->setMotionPlanRequest(planning_scene_,request_, stomp_config_,res.error_code_))
      {
        res.error_code_.val = moveit_msgs::MoveItErrorCodes::FAILURE;
        return false;
      }
      stomp_->setConfig(stomp_config_);
      planning_success = stomp_->solve(start,goal,parameters);
    }
  }

  // stopping timer
  timeout_timer.stop();

  // Handle results
  if(planning_success)
  {
    if(!parametersToJointTrajectory(parameters,trajectory))
    {
      res.error_code_.val = moveit_msgs::MoveItErrorCodes::PLANNING_FAILED;
      return false;
    }

    // creating request response
    moveit::core::RobotState robot_state = planning_scene_->getCurrentState();
    moveit::core::robotStateMsgToRobotState(request_.start_state,robot_state);
    res.trajectory_[0]= robot_trajectory::RobotTrajectoryPtr(new robot_trajectory::RobotTrajectory(
        robot_model_,group_));
    res.trajectory_.back()->setRobotTrajectoryMsg( robot_state,trajectory);
  }
  else
  {
    res.error_code_.val = moveit_msgs::MoveItErrorCodes::PLANNING_FAILED;
    return false;
  }

  // checking against planning scene
  if(planning_scene_ && !planning_scene_->isPathValid(*res.trajectory_.back(),group_,true))
  {
    res.error_code_.val = moveit_msgs::MoveItErrorCodes::PLANNING_FAILED;
    success = false;
    ROS_ERROR_STREAM("STOMP Trajectory is in collision");
  }

  ros::WallDuration wd = ros::WallTime::now() - start_time;
  res.processing_time_[0] = ros::Duration(wd.sec, wd.nsec).toSec();
  ROS_INFO_STREAM("STOMP found a valid path after "<<res.processing_time_[0]<<" seconds");

  return true;
}

bool StompPlanner::getSeedParameters(Eigen::MatrixXd& parameters) const
{
  using namespace utils::kinematics;
  using namespace utils::polynomial;

  auto within_tolerance = [&](const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol) -> bool
  {
    double dist = (a - b).cwiseAbs().sum();
    return dist <= tol;
  };

  trajectory_msgs::JointTrajectory traj;
  if(!extractSeedTrajectory(request_,traj))
  {
    ROS_DEBUG("%s Found no seed trajectory",getName().c_str());
    return false;
  }

  if(!jointTrajectorytoParameters(traj,parameters))
  {
    ROS_ERROR("%s Failed to created seed parameters from joint trajectory",getName().c_str());
    return false;
  }

  if(parameters.cols()<= 2)
  {
    ROS_ERROR("%s Found less than 3 points in seed trajectory",getName().c_str());
    return false;
  }

  /* ********************************************************************************
   * Validating seed trajectory by ensuring that it does obey the
   * motion plan request constraints
   */
  moveit::core::RobotState state = planning_scene_->getCurrentState();
  const auto* group = state.getJointModelGroup(group_);
  const auto& joint_names = group->getActiveJointModelNames();
  const auto& tool_link = group->getLinkModelNames().back();
  Eigen::VectorXd start, goal;

  // We check to see if the start state in the request and the seed state are 'close'
  if (moveit::core::robotStateMsgToRobotState(request_.start_state, state))
  {

    if(!state.satisfiesBounds(group))
    {
      ROS_ERROR("Start state is out of bounds");
      return false;
    }

    // copying start joint values
    start.resize(joint_names.size());
    for(auto j = 0u; j < joint_names.size(); j++)
    {
      start(j) = state.getVariablePosition(joint_names[j]);
    }

    if(within_tolerance(parameters.leftCols(1),start,MAX_START_DISTANCE_THRESH))
    {
      parameters.leftCols(1) = start;
    }
    else
    {
      ROS_ERROR("%s Start State is in discrepancy with the seed trajectory",getName().c_str());
      return false;
    }
  }
  else
  {
    ROS_ERROR("%s Failed to get start state joints",getName().c_str());
    return false;
  }

  // We now extract the goal and make sure that the seed's goal obeys the goal constraints
  bool found_goal = false;
  goal = parameters.rightCols(1); // initializing goal;
  for(auto& gc : request_.goal_constraints)
  {
    if(!gc.joint_constraints.empty())
    {
      // copying goal values into state
      for(auto j = 0u; j < gc.joint_constraints.size() ; j++)
      {
        auto jc = gc.joint_constraints[j];
        state.setVariablePosition(jc.joint_name,jc.position);
      }

      // copying values into goal array
      if(!state.satisfiesBounds(group))
      {
        ROS_ERROR("%s Requested Goal joint pose is out of bounds",getName().c_str());
        continue;
      }

      for(auto j = 0u; j < joint_names.size(); j++)
      {
        goal(j) = state.getVariablePosition(joint_names[j]);
      }

      found_goal = true;
      break;
    }

    // now check Cartesian constraint
    state.updateLinkTransforms();
    Eigen::Affine3d start_tool_pose = state.getGlobalLinkTransform(tool_link);
    boost::optional<moveit_msgs::Constraints> tool_constraints = curateCartesianConstraints(gc,start_tool_pose);
    if(!tool_constraints.is_initialized())
    {
      ROS_WARN("Cartesian Goal could not be created from provided constraints");
      found_goal = true;
      break;
    }

    Eigen::VectorXd solution;
    ik_solver_->setKinematicState(state);
    if(ik_solver_->solve(goal,tool_constraints.get(),solution))
    {
      goal = solution;
      found_goal = true;
      break;
    }
    else
    {
      ROS_ERROR("A valid ik solution for the given Cartesian constraints was not found ");
      ROS_DEBUG_STREAM_NAMED(DEBUG_NS,"IK failed with goal constraint \n"<<tool_constraints.get());
      ROS_DEBUG_STREAM_NAMED(DEBUG_NS,"Reference Tool pose used was: \n"<<start_tool_pose.matrix());
    }
  }

  // forcing the goal into the seed trajectory
  if(found_goal)
  {
    if(within_tolerance(parameters.rightCols(1),goal,MAX_START_DISTANCE_THRESH))
    {
      parameters.rightCols(1) = goal;
    }
    else
    {
      ROS_ERROR("%s Goal in seed is too far away from requested goal constraints",getName().c_str());
      return false;
    }
  }
  else
  {
    ROS_ERROR("%s requested goal constraint was invalid or unreachable, comparison with goal in seed isn't possible",getName().c_str());
    return false;
  }

  if(!applyPolynomialSmoothing(robot_model_,group_,parameters,5,1e-5))
  {
    return false;
  }

  return true;
}

bool StompPlanner::parametersToJointTrajectory(const Eigen::MatrixXd& parameters,
                                               trajectory_msgs::JointTrajectory& trajectory)
{
  // filling trajectory joint values
  trajectory.joint_names = robot_model_->getJointModelGroup(group_)->getActiveJointModelNames();
  trajectory.points.clear();
  trajectory.points.resize(parameters.cols());
  std::vector<double> vals(parameters.rows());
  std::vector<double> zeros(parameters.rows(),0.0);
  for(auto t = 0u; t < parameters.cols() ; t++)
  {
    Eigen::VectorXd::Map(&vals[0],vals.size()) = parameters.col(t);
    trajectory.points[t].positions = vals;
    trajectory.points[t].velocities = zeros;
    trajectory.points[t].accelerations = zeros;
    trajectory.points[t].time_from_start = ros::Duration(0.0);
  }

  trajectory_processing::IterativeParabolicTimeParameterization time_generator;
  robot_trajectory::RobotTrajectory traj(robot_model_,group_);
  moveit::core::RobotState robot_state = planning_scene_->getCurrentState();
  if(!moveit::core::robotStateMsgToRobotState(request_.start_state,robot_state))
  {
    return false;
  }

  traj.setRobotTrajectoryMsg(robot_state,trajectory);

  // converting to msg
  moveit_msgs::RobotTrajectory robot_traj_msgs;
  if(time_generator.computeTimeStamps(traj,request_.max_velocity_scaling_factor))
  {
    traj.getRobotTrajectoryMsg(robot_traj_msgs);
    trajectory = robot_traj_msgs.joint_trajectory;
  }
  else
  {
    ROS_ERROR("%s Failed to generate timing data",getName().c_str());
    return false;
  }
  return true;
}

bool StompPlanner::jointTrajectorytoParameters(const trajectory_msgs::JointTrajectory& traj, Eigen::MatrixXd& parameters) const
{
  const auto dof = traj.joint_names.size();
  const auto timesteps = traj.points.size();

  Eigen::MatrixXd mat (dof, timesteps);

  for (size_t step = 0; step < timesteps; ++step)
  {
    for (size_t joint = 0; joint < dof; ++joint)
    {
      mat(joint, step) = traj.points[step].positions[joint];
    }
  }

  parameters = mat;
  return true;
}

bool StompPlanner::extractSeedTrajectory(const moveit_msgs::MotionPlanRequest& req, trajectory_msgs::JointTrajectory& seed) const
{
  if (req.trajectory_constraints.constraints.empty())
    return false;

  const auto* joint_group = robot_model_->getJointModelGroup(group_);
  const auto& names = joint_group->getActiveJointModelNames();
  const auto dof = names.size();

  const auto& constraints = req.trajectory_constraints.constraints; // alias to keep names short
  // Test the first point to ensure that it has all of the joints required
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    auto n = constraints[i].joint_constraints.size();
    if (n != dof) // first test to ensure that dimensionality is correct
    {
      ROS_WARN("Seed trajectory index %lu does not have %lu constraints (has %lu instead).", i, dof, n);
      return false;
    }

    trajectory_msgs::JointTrajectoryPoint joint_pt;

    for (size_t j = 0; j < constraints[i].joint_constraints.size(); ++j)
    {
      const auto& c = constraints[i].joint_constraints[j];
      if (c.joint_name != names[j])
      {
        ROS_WARN("Seed trajectory (index %lu, joint %lu) joint name '%s' does not match expected name '%s'",
                 i, j, c.joint_name.c_str(), names[j].c_str());
        return false;
      }
      joint_pt.positions.push_back(c.position);
    }

    seed.points.push_back(joint_pt);
  }

  seed.joint_names = names;
  return true;
}

moveit_msgs::TrajectoryConstraints StompPlanner::encodeSeedTrajectory(const trajectory_msgs::JointTrajectory &seed)
{
  ROS_INFO_STREAM("encodeSeedTrajectory関数を使用します");

  moveit_msgs::TrajectoryConstraints res;

  const auto dof = seed.joint_names.size();

  for (size_t i = 0; i < seed.points.size(); ++i) // for each time step
  {
    moveit_msgs::Constraints c;

    if (seed.points[i].positions.size() != dof)
      throw std::runtime_error("All trajectory position fields must have same dimensions as joint_names");

    for (size_t j = 0; j < dof; ++j) // for each joint
    {
      moveit_msgs::JointConstraint jc;
      jc.joint_name = seed.joint_names[j];
      jc.position = seed.points[i].positions[j];

      c.joint_constraints.push_back(jc);
    }

    res.constraints.push_back(std::move(c));
  }

  return res;
}

bool StompPlanner::getStartAndGoal(Eigen::VectorXd& start, Eigen::VectorXd& goal)
{
  using namespace moveit::core;
  using namespace utils::kinematics;

  RobotStatePtr state(new RobotState(planning_scene_->getCurrentState()));
  const JointModelGroup* joint_group = robot_model_->getJointModelGroup(group_);
  std::string tool_link = joint_group->getLinkModelNames().back();
  bool found_goal = false;

  try
  {
    // copying start state
    if(!robotStateMsgToRobotState(request_.start_state,*state))
    {
      ROS_ERROR("%s Failed to extract start state from MotionPlanRequest",getName().c_str());
      return false;
    }

    // copying start joint values
    const std::vector<std::string> joint_names= state->getJointModelGroup(group_)->getActiveJointModelNames();
    start.resize(joint_names.size());
    goal.resize(joint_names.size());

    if(!state->satisfiesBounds(state->getJointModelGroup(group_)))
    {
      ROS_ERROR("%s Start joint pose is out of bounds",getName().c_str());
      return false;
    }

    for(auto j = 0u; j < joint_names.size(); j++)
    {
      start(j) = state->getVariablePosition(joint_names[j]);
    }

    // check goal constraint
    if(request_.goal_constraints.empty())
    {
      ROS_ERROR("%s A goal constraint was not provided",getName().c_str());
      return false;
    }

    // extracting goal joint values
    for(const auto& gc : request_.goal_constraints)
    {

      // check joint constraints first
      if(!gc.joint_constraints.empty())
      {

        // copying goal values into state
        for(auto j = 0u; j < gc.joint_constraints.size() ; j++)
        {
          auto jc = gc.joint_constraints[j];
          state->setVariablePosition(jc.joint_name,jc.position);
        }


        if(!state->satisfiesBounds(state->getJointModelGroup(group_)))
        {
          ROS_ERROR("%s Requested Goal joint pose is out of bounds",getName().c_str());
          continue;
        }

        ROS_DEBUG("%s Found goal from joint constraints",getName().c_str());

        // copying values into goal array
        for(auto j = 0u; j < joint_names.size(); j++)
        {
          goal(j) = state->getVariablePosition(joint_names[j]);
        }

        found_goal = true;
        break;

      }

      // now check cartesian constraint
      state->updateLinkTransforms();
      Eigen::Affine3d start_tool_pose = state->getGlobalLinkTransform(tool_link);
      boost::optional<moveit_msgs::Constraints> tool_constraints = curateCartesianConstraints(gc,start_tool_pose);
      if(!tool_constraints.is_initialized())
      {
        ROS_WARN("Cartesian Goal could not be created from provided constraints");
        found_goal = true;
        break;
      }

      // now solve ik
      Eigen::VectorXd solution;
      Eigen::VectorXd seed = start;
      ik_solver_->setKinematicState(*state);
      if(ik_solver_->solve(seed,tool_constraints.get(),solution))
      {
        goal = solution;
        found_goal = true;
        break;
      }
      else
      {
        ROS_ERROR("A valid ik solution for the given Cartesian constraints was not found ");
        ROS_DEBUG_STREAM_NAMED(DEBUG_NS,"IK failed with goal constraint \n"<<tool_constraints.get());
        ROS_DEBUG_STREAM_NAMED(DEBUG_NS,"Reference Tool pose used was: \n"<<start_tool_pose.matrix());
      }
    }

    ROS_ERROR_COND(!found_goal,"%s was unable to retrieve the goal from the MotionPlanRequest",getName().c_str());

  }
  catch(moveit::Exception &e)
  {
    ROS_ERROR("Failure retrieving start or goal state joint values from request %s", e.what());
    return false;
  }

  return found_goal;
}


bool StompPlanner::canServiceRequest(const moveit_msgs::MotionPlanRequest &req) const
{
  // check group
  if(req.group_name != getGroupName())
  {
    ROS_ERROR("STOMP: Unsupported planning group '%s' requested", req.group_name.c_str());
    return false;
  }

  // check for single goal region
  if (req.goal_constraints.size() != 1)
  {
    ROS_ERROR("STOMP: Can only handle a single goal region.");
    return false;
  }

  // check that we have joint or cartesian constraints at the goal
  const auto& gc = req.goal_constraints[0];
  if ((gc.joint_constraints.size() == 0) &&
      !utils::kinematics::isCartesianConstraints(gc))
  {
    ROS_ERROR("STOMP couldn't find either a joint or cartesian goal.");
    return false;
  }

  return true;
}

bool StompPlanner::terminate()
{
  if(stomp_)
  {
    if(!stomp_->cancel())
    {
      ROS_ERROR_STREAM("Failed to interrupt Stomp");
      return false;
    }
  }
  return true;
}

void StompPlanner::clear()
{
  stomp_->clear();
}

bool StompPlanner::getConfigData(ros::NodeHandle &nh, std::map<std::string, XmlRpc::XmlRpcValue> &config, std::string param)
{
  // Create a stomp planner for each group
  XmlRpc::XmlRpcValue stomp_config;
  if(!nh.getParam(param, stomp_config))
  {
    ROS_ERROR("The 'stomp' configuration parameter was not found");
    return false;
  }

  // each element under 'stomp' should be a group name
  std::string group_name;
  try
  {
    for(XmlRpc::XmlRpcValue::iterator v = stomp_config.begin(); v != stomp_config.end(); v++)
    {
      group_name = static_cast<std::string>(v->second["group_name"]);
      config.insert(std::make_pair(group_name, v->second));
    }
    return true;
  }
  catch(XmlRpc::XmlRpcException& e )
  {
    ROS_ERROR("Unable to parse ROS parameter:\n %s",stomp_config.toXml().c_str());
    return false;
  }
}


} /* namespace stomp_moveit_interface */
