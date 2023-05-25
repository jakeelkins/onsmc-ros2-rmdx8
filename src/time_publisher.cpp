#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "onsmc_servo/msg/time.hpp"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

class TimePublisher : public rclcpp::Node{
public:

    bool go_flag = false;
    float t = 0.0f;  // actual t value we publish

    // nominal control period (s)
    float control_period = 0.001;

    // needed for our timer: ms duration cast
    std::chrono::duration<long double, std::milli> control_period_ms = control_period*1000ms;

    TimePublisher():Node("time_publisher"), count_(0){
        // subscribe to "go" topic (bool)
        subscription_ = this->create_subscription<std_msgs::msg::Bool>(
        "go", 10, std::bind(&TimePublisher::go_callback, this, std::placeholders::_1));

        // publish to time topic "t"
        publisher_ = this->create_publisher<onsmc_servo::msg::Time>("t", 10);

        timer_ = this->create_wall_timer(
        control_period_ms, std::bind(&TimePublisher::timer_callback, this));
    }

private:

    size_t count_;  // TODO: what is this used for ?

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<onsmc_servo::msg::Time>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription_;

    void timer_callback(){
        auto message = onsmc_servo::msg::Time();

        // timer is zero until it gets the "go" command.
        if (go_flag){
            t += control_period;
        }

        message.curr_t = t;
        //RCLCPP_INFO(this->get_logger(), "Publishing: %f", message.curr_t);
        publisher_->publish(message);
    }

    void go_callback(const std_msgs::msg::Bool & msg){
        // sets the flag to true
        if (msg.data){
            RCLCPP_INFO_ONCE(this->get_logger(), "Go flag received: experiment start.");
            go_flag = true;
        }
    }

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TimePublisher>());
  rclcpp::shutdown();
  return 0;
}
