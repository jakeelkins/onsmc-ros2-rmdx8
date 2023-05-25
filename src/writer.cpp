#include <functional>
#include <memory>
#include <math.h>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "onsmc_servo/msg/time.hpp"
#include "onsmc_servo/msg/desired_trajectory.hpp"
#include "onsmc_servo/msg/state.hpp"
#include "onsmc_servo/msg/control.hpp"

using std::placeholders::_1;
using namespace std;

// init save arrays
vector<float> t_save;
vector<float> x_save;
vector<float> xd_save;
vector<float> u_save;
vector<float> I_save;

class Writer : public rclcpp::Node {
public:

    Writer():Node("writer"){

        // subscribe to "go" topic (bool)
        subscription_go = this->create_subscription<std_msgs::msg::Bool>(
            "go", 10, std::bind(&Writer::go_callback, this, std::placeholders::_1));

        // subscribe to topic t as input
        t_subscription = this->create_subscription<onsmc_servo::msg::Time>(
            "t", 10, std::bind(&Writer::t_received_callback, this, _1));

        // subscribe to desired trajectory
        subscription_xd = this->create_subscription<onsmc_servo::msg::DesiredTrajectory>(
            "desired_trajectory", 10, std::bind(&Writer::traj_received_callback, this, _1));

        subscription_x = this->create_subscription<onsmc_servo::msg::State>(
            "state", 10, std::bind(&Writer::state_received_callback, this, _1));

        // subscribe to control topic
        subscription_u = this->create_subscription<onsmc_servo::msg::Control>(
            "control", 10, std::bind(&Writer::control_received_callback, this, _1));

    }

private:

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription_go;
  rclcpp::Subscription<onsmc_servo::msg::Time>::SharedPtr t_subscription;
  rclcpp::Subscription<onsmc_servo::msg::DesiredTrajectory>::SharedPtr subscription_xd;
  rclcpp::Subscription<onsmc_servo::msg::State>::SharedPtr subscription_x;
  rclcpp::Subscription<onsmc_servo::msg::Control>::SharedPtr subscription_u;

  bool go_flag = false;
  // current values for record callback
  float t;

  float qd;
  float qd_dot;
  float qd_ddot;

  float q;
  float q_dot;

  float u;

  float I;

  void go_callback(const std_msgs::msg::Bool &msg){
        // sets the flag to true
        if (msg.data){
            //RCLCPP_INFO_ONCE(this->get_logger(), "Go flag received: experiment start.");
            go_flag = true;
        }
  }

  void t_received_callback(const onsmc_servo::msg::Time &msg){

    t = msg.curr_t;

    //RCLCPP_INFO(this->get_logger(), "Current t: %f s", msg.curr_t);

  }

  void traj_received_callback(const onsmc_servo::msg::DesiredTrajectory &desired_traj_msg){

    qd = desired_traj_msg.qd;
    qd_dot = desired_traj_msg.qd_dot;
    qd_ddot = desired_traj_msg.qd_ddot;

  }

  void state_received_callback(const onsmc_servo::msg::State &state_msg){

    // -- record current state inputs --
    q = state_msg.q;
    q_dot = state_msg.q_dot;
    I = state_msg.curr_i;

  }

  void control_received_callback(const onsmc_servo::msg::Control &control_msg){

    u = control_msg.u;

    if (go_flag){
        // record when we get control
        t_save.push_back(t);
        u_save.push_back(u);
        x_save.push_back(q);
        xd_save.push_back(qd);
        I_save.push_back(I);
    }

  }
  
};

void write_csv(string filename, vector<std::pair<string, vector<float>>> dataset){
    // Make a CSV file with one or more columns of float values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<float>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    ofstream myFile(filename);
    
    // Send column names to the stream
    for(int j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset.at(j).first;
        if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
    }
    myFile << "\n";
    
    // Send data to the stream
    for(int i = 0; i < dataset.at(0).second.size(); ++i)
    {
        for(int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).second.at(i);
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    
    // Close the file
    myFile.close();
}


int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);

    std::shared_ptr<rclcpp::Node> node = std::make_shared<Writer>();

    rclcpp::spin(node);
    rclcpp::shutdown();

    // wrap and save
    vector<std::pair<string, vector<float>>> out_dataset = {{"t", t_save},
                    {"u", u_save}, {"xd", xd_save}, {"x", x_save}, {"I", I_save}};

    // Write the vector to CSV
    // i have no idea how to do relative paths in cpp
    printf("writing %i rows of data out ...", out_dataset.size());
    write_csv("/home/jake/ros2_ws/src/onsmc_servo/data/sim_out.csv", out_dataset);
    return 0;
}
