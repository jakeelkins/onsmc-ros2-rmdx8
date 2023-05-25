#include <functional>
#include <memory>
#include <math.h>

#include "rclcpp/rclcpp.hpp"
#include "onsmc_servo/msg/time.hpp"
#include "onsmc_servo/msg/desired_trajectory.hpp"

using std::placeholders::_1;

class TrajectoryGenerator : public rclcpp::Node {
public:

  TrajectoryGenerator():Node("trajectory_generator"){

    // subscribe to topic t as input
    subscription_ = this->create_subscription<onsmc_servo::msg::Time>(
      "t", 10, std::bind(&TrajectoryGenerator::t_received_callback, this, _1));

    // publish to desired_trajectory topic
    publisher_ = this->create_publisher<onsmc_servo::msg::DesiredTrajectory>("desired_trajectory", 10);

  }

private:

  rclcpp::Subscription<onsmc_servo::msg::Time>::SharedPtr subscription_;
  rclcpp::Publisher<onsmc_servo::msg::DesiredTrajectory>::SharedPtr publisher_;

  void t_received_callback(const onsmc_servo::msg::Time &msg){

    auto desired_traj_msg = onsmc_servo::msg::DesiredTrajectory();
    float t = msg.curr_t;

    //RCLCPP_INFO(this->get_logger(), "Current t: %f s", msg.curr_t);

    // -- desired trajectory calcs --
    // float qd = 1 - cos(t);
    // float qd_dot = sin(t);
    // float qd_ddot = cos(t);

    float qd = sin(t/10);
    float qd_dot = cos(t/10)/10;
    float qd_ddot = -sin(t/10)/100;

    // float qd = 1;
    // float qd_dot = 0.0f;
    // float qd_ddot = 0.0f;

    // float qd = 0.0f;
    // float qd_dot = 0.0f;
    // float qd_ddot = 0.0f;

    // if (t > 2.0f){
    //   //qd = 30*(3.14159265/180);
    //   qd = 0.3;
    // }
    
    // if (t > 2.2f) {
    //   qd = 0.0f;
    // }

    // if (t > 3.0f) {
    //   qd = -0.3f;
    // }

    // if (t > 3.2f) {
    //   qd = 0.0f;
    // }

    desired_traj_msg.qd = qd;
    desired_traj_msg.qd_dot = qd_dot;
    desired_traj_msg.qd_ddot = qd_ddot;

    //RCLCPP_INFO(this->get_logger(), "Publishing desired trajectory...");

    publisher_->publish(desired_traj_msg);

  }
  
};

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);

  std::shared_ptr<rclcpp::Node> node = std::make_shared<TrajectoryGenerator>();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
