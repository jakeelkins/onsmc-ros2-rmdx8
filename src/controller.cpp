#include <chrono>
#include <functional>
#include <memory>
#include <math.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
// #include <string>
// #include <fstream>
// #include <vector>
// #include <utility> // std::pair

#include "rclcpp/rclcpp.hpp"
#include "onsmc_servo/msg/desired_trajectory.hpp"
#include "onsmc_servo/msg/control.hpp"
#include "onsmc_servo/msg/state.hpp"

#include "onsmc_servo/onsmc.h"

using namespace std::chrono_literals;
using std::placeholders::_1;
using namespace std;

class Controller: public rclcpp::Node {
public:

  // nominal control frequency (s) CURRENT: ensure same as time publisher
  float control_period = 0.001;

  // needed for our timer: ms duration cast
  chrono::duration<long double, std::milli> control_period_ms = control_period*1000ms;


  Controller()
  :Node("controller"),
  onsmc(input_dim, output_dim, control_period)
  {

    // populate vector ICs in constructor:
    vector<float> _y (
        output_dim,
        0.0f
    );

    vector<float> _y_dot (
        output_dim,
        0.0f
    );

    vector<float> _yd (
        output_dim,
        0.0f
    );

    vector<float> _yd_dot (
        output_dim,
        0.0f
    );

    vector<float> _yd_ddot (
        output_dim,
        0.0f
    );

    // variables needed
    vector<float> _u (
        output_dim,
        0.0f
    );

    y = _y;
    y_dot = _y_dot;
    
    yd = _yd;
    yd_dot = _yd_dot;
    yd_ddot = _yd_ddot;

    u = _u;

    // subscribe to current state x as input
    subscription_x = this->create_subscription<onsmc_servo::msg::State>(
      "state", 10, std::bind(&Controller::state_receive_callback, this, _1));

    // subscribe to topic desired_trajectory as input
    subscription_xd = this->create_subscription<onsmc_servo::msg::DesiredTrajectory>(
      "desired_trajectory", 10, std::bind(&Controller::traj_receive_callback, this, _1));

    // publish to control topic
    publisher_ = this->create_publisher<onsmc_servo::msg::Control>("control", 10);

    // wall timer for publishing control
    timer_ = this->create_wall_timer(
        control_period_ms, std::bind(&Controller::control_callback, this));

  }

private:

  rclcpp::Subscription<onsmc_servo::msg::State>::SharedPtr subscription_x;
  rclcpp::Subscription<onsmc_servo::msg::DesiredTrajectory>::SharedPtr subscription_xd;
  rclcpp::Publisher<onsmc_servo::msg::Control>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  unsigned int input_dim = 7;
  unsigned int output_dim = 1;

  // ONSMC controller
  ONSMC onsmc;

  // --  state ICs --
  // match these to the desireds so u is 0 before the experiment runs
  float q = 0.0f;
  float q_dot = 0.0f;
  float I = 0.0f;

  // -- desired trajectory ICs --
  float qd = 0.0f;
  float qd_dot = 0.0f;
  float qd_ddot = 0.0f;

  // use vectors
  vector<float> y;
  vector<float> y_dot;

  vector<float> yd;
  vector<float> yd_dot;
  vector<float> yd_ddot;

  vector<float> u;


  void state_receive_callback(const onsmc_servo::msg::State &state_msg){

    // -- record current state inputs --
    q = state_msg.q;
    q_dot = state_msg.q_dot;
    I = state_msg.curr_i;

    // put the state values into the vectors
    y[0] = q;
    y_dot[0] = q_dot; 

  }

  void traj_receive_callback(const onsmc_servo::msg::DesiredTrajectory &desired_traj_msg){

    // -- record desired trajectory inputs --
    qd = desired_traj_msg.qd;
    qd_dot = desired_traj_msg.qd_dot;
    qd_ddot = desired_traj_msg.qd_ddot;

    // put the desireds in the vector
    yd[0] = qd;
    yd_dot[0] = qd_dot;
    yd_ddot[0] = qd_ddot;

  }

  // THE MEAT: controller.
  void control_callback(){

    // out msg:
    auto control_msg = onsmc_servo::msg::Control();

    //RCLCPP_INFO(this->get_logger(), "curr q: %f  curr q_dot: %f  curr qd: %f", q, q_dot, qd);
    //RCLCPP_INFO_ONCE(this->get_logger(), "input_dim: %i  hidden_dim: %i  output_dim: %i", onsmc.input_dim, onsmc.hidden_dim, onsmc.output_dim);
    
    // ---- control calcs ----
    // get control: puts it into u vector.
    onsmc.get_control(u.data(), y.data(), y_dot.data(),
                      yd.data(), yd_dot.data(), yd_ddot.data());

    // scale
    u[0] = 0.12*u[0];

    // deadzone inverse
    //float dz = 0.335f;
    float dz = 0.0f;

    if (u[0] > 0.0f){
      u[0] = u[0] + dz;
    } else if (u[0] < 0.0f){
      u[0] = u[0] - dz;
    }

    

    // clip
    float clip = 3.0f;

    if (u[0] > clip){
      u[0] = clip;
    } else if (u[0] < -clip){
      u[0] = -clip;
    }

    // ------------------------

    control_msg.u = u[0];

    publisher_->publish(control_msg);

    if (qd > 0.9){
      RCLCPP_INFO_ONCE(this->get_logger(), "W[0]: %f", onsmc.NN.W[0]);
    }
    
  }
  
};

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);

  std::shared_ptr<rclcpp::Node> node = std::make_shared<Controller>();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}