#include <chrono>
#include <functional>
#include <memory>
#include <math.h>
#include <cstring>

#include <net/if.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include <linux/can.h>
#include <linux/can/raw.h>

#include "rclcpp/rclcpp.hpp"
#include "onsmc_servo/msg/state.hpp"

using namespace std::chrono_literals;

const double pi = 3.14159265358979323846;

class Reader : public rclcpp::Node {
public:

    // nominal state update period (s)
    float update_period = 0.001;

    // needed for our timer: ms duration cast
    std::chrono::duration<long double, std::milli> update_period_ms = update_period*1000ms;


    // ------ for servo -----
    int s;
    int stat;
    struct sockaddr_can addr;
    struct ifreq ifr;

    struct can_frame angle_frame_out;
    struct can_frame status_frame_out;

    struct can_frame frame_in;

    int nbytes;
    socklen_t len;

    int servo_id = 0x141;
    int can_dlc = 8;
    int gear_ratio = 6;


    Reader():Node("reader"){

        // --- cmnds
        angle_frame_out.can_id = servo_id;
        angle_frame_out.can_dlc = can_dlc;

        // read multi_turn_angle cmnd
        angle_frame_out.data[0] = 0x92;
        angle_frame_out.data[1] = 0x00;
        angle_frame_out.data[2] = 0x00;
        angle_frame_out.data[3] = 0x00;
        angle_frame_out.data[4] = 0x00;
        angle_frame_out.data[5] = 0x00;
        angle_frame_out.data[6] = 0x00;
        angle_frame_out.data[7] = 0x00;

        status_frame_out.can_id = servo_id;
        status_frame_out.can_dlc = can_dlc;

        // read motor status 2 cmnd
        status_frame_out.data[0] = 0x9C;
        status_frame_out.data[1] = 0x00;
        status_frame_out.data[2] = 0x00;
        status_frame_out.data[3] = 0x00;
        status_frame_out.data[4] = 0x00;
        status_frame_out.data[5] = 0x00;
        status_frame_out.data[6] = 0x00;
        status_frame_out.data[7] = 0x00;

        // --- connecting to servo ---
        connect();

        if (stat){
            disconnect();
        } else {
            //printf("Socket connected. \n");
            RCLCPP_INFO(this->get_logger(), "Reader connected to socket");
        }

        // --- ros stuff ---

        // publishes to state topic
        publisher_ = this->create_publisher<onsmc_servo::msg::State>("state", 10);

        // periodic state reader
        timer_ = this->create_wall_timer(
        update_period_ms, std::bind(&Reader::timer_callback, this));

    }


    // ---- servo functions ----

    void connect(){
        // connect and bind to the servo

        s = -1;
        stat = EXIT_SUCCESS;
        len = sizeof(addr);

        s = socket(PF_CAN, SOCK_RAW, CAN_RAW);

        if (s < 0){
            perror("Error while creating socket");
            stat = EXIT_FAILURE;
        }

        // bind
    	/* get interface index */
        strcpy(ifr.ifr_name, "slcan0");
        ioctl(s, SIOCGIFINDEX, &ifr);

        // NOTE: this has to come after the strcpy and ioctl.
        // this gave me fits.
        addr.can_family  = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        
        /* bind CAN socket to the given interface */
        int bind_stat = bind(s, (struct sockaddr *)&addr, sizeof(addr));

        if (bind_stat < 0){
            perror("Error in socket bind");
            stat = EXIT_FAILURE;
        }

        //return stat;
    }

    void disconnect(){
        // disconnect when something happens
        if (s >= 0){
            close(s);
            //printf("Socket disconnected. \n");
            RCLCPP_INFO(this->get_logger(), "Reader disconnected from socket");
        }
    }

    // commands

    void read_angle(){

        // puts the current angle (in radians) into variable curr_angle

        //RCLCPP_INFO(this->get_logger(), "sending update cmnd..");

        nbytes = sendto(s, &angle_frame_out, sizeof(struct can_frame),
                0, (struct sockaddr*)&addr, sizeof(addr));

        if (nbytes <= 0){
            //perror("Error in read angle send");
            stat = EXIT_FAILURE;
        }

        for(;;){
            // need a while loop to ensure we are filtering the
            // right angle command reply (not the torque command replies)

            //RCLCPP_INFO(this->get_logger(), "waiting on angle reply..");

            nbytes = recvfrom(s, &frame_in, sizeof(struct can_frame),
                    0, (struct sockaddr*)&addr, &len);

            //RCLCPP_INFO(this->get_logger(), "got some angle reply..");

            if (nbytes <= 0){
                //perror("Read zero bytes in read angle");
                stat = EXIT_FAILURE;
                break;
            }

            if (frame_in.data[0] == 0x92){
                // got our command reply, report
                //RCLCPP_INFO(this->get_logger(), "got the correct angle reply");

                // now need to convert angle from int64 little-endian, 7 bytes, to float.
                // also comes out in degrees.
                unsigned char angle_bytes[7]{frame_in.data[1],
                                            frame_in.data[2],
                                            frame_in.data[3],
                                            frame_in.data[4],
                                            frame_in.data[5],
                                            frame_in.data[6],
                                            frame_in.data[7]};

                int angle_int;
                std::memcpy(&angle_int, angle_bytes, sizeof(int));

                // angle_int now holds angle in degrees to 0.01 LSB:
                q = angle_int*0.01f*pi/180.0f/gear_ratio;

                break;
            } else {
                nbytes = sendto(s, &angle_frame_out, sizeof(struct can_frame),
                0, (struct sockaddr*)&addr, sizeof(addr));

                if (nbytes <= 0){
                    //perror("Error in read angle send");
                    stat = EXIT_FAILURE;
                    //break;
                }
            };
        }



        //return stat;
    }


    void get_motor_status(){

        // this command gets lots of motor info (temp, voltage, speed, encoder pos)
        // we just use it to get motor speed (velocity to be precise)

        //RCLCPP_INFO(this->get_logger(), "sending status cmnd..");

        nbytes = sendto(s, &status_frame_out, sizeof(struct can_frame),
                0, (struct sockaddr*)&addr, sizeof(addr));

        if (nbytes <= 0){
            //perror("Error in get motor status send");
            stat = EXIT_FAILURE;
        }

        for(;;){
            // need a while loop to ensure we are filtering the
            // right command reply (not the torque command replies)

            //RCLCPP_INFO(this->get_logger(), "waiting on status cmnd..");

            nbytes = recvfrom(s, &frame_in, sizeof(struct can_frame),
                    0, (struct sockaddr*)&addr, &len);

            //RCLCPP_INFO(this->get_logger(), "got a status cmnd..");

            if (nbytes <= 0){
                //perror("Read zero bytes in get motor status");
                stat = EXIT_FAILURE;
                break;
            }

            if (frame_in.data[0] == 0x9C){
                // got our command reply, report
                //RCLCPP_INFO(this->get_logger(), "got correct status cmnd");
                // now need to convert speed from int16 little-endian, 2 bytes, to float.
                // also comes out in degrees.
                unsigned char velo_bytes[2]{frame_in.data[4],
                                            frame_in.data[5]};

                short velo_short;
                std::memcpy(&velo_short, velo_bytes, sizeof(short));

                // velo_int now holds speed in degrees/s to 1 LSB:
                q_dot = (float)velo_short*pi/180.0f/gear_ratio;  // rad/s

                // read current
                unsigned char current_bytes[2]{frame_in.data[2],
                                            frame_in.data[3]};

                short current_short;
                std::memcpy(&current_short, current_bytes, sizeof(short));
                current_I = (float)current_short;

                break;
            } else {
                nbytes = sendto(s, &status_frame_out, sizeof(struct can_frame),
                0, (struct sockaddr*)&addr, sizeof(addr));

                if (nbytes <= 0){
                    //perror("Error in get motor status send");
                    stat = EXIT_FAILURE;
                    //break;
                }
            };
        }

        

        //return stat;
    }

private:

    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<onsmc_servo::msg::State>::SharedPtr publisher_;

    // READER ICs
    float q = 0.0f;
    float q_dot = 0.0f;
    float current_I = 0.0f;

    void timer_callback(){

        // ----- update q, q_dot -----
        read_angle();

        get_motor_status();
        // --- put in message, publish ---

        auto state_msg = onsmc_servo::msg::State();

        state_msg.q = q;
        state_msg.q_dot = q_dot;
        state_msg.curr_i = current_I;

        //RCLCPP_INFO(this->get_logger(), "publishing curr q: %f  curr q_dot: %f", q, q_dot);

        publisher_->publish(state_msg);

    }
  
};

int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);

    std::shared_ptr<rclcpp::Node> node = std::make_shared<Reader>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
