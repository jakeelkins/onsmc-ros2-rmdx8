# ONSMC in ROS2 on the RMD X8

This repo implements my most recent paper, "Online Neural Sldiing Mode Control with Guaranteed Stability", on the RMD X8 servo in our lab using ros2.

The only difference in the previous ROS graph is linking my custom header files for ONSMC (it drastically cleaned up the code). You need to make sure the terminal that you source ROS in has the folder ~/ros2_ws/build/onsmc_servo/ in its $LD_LIBRARY_PATH env variable:

    echo $LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/build
    export LD_LIBRARY_PATH

Setup is the same. If you are just powering up the servo, you need to run

    sudo slcand -c -o -s8 -t hw -S 3000000 /dev/ttyUSB0
    sudo ip link set up slcan0

Then, you should see the servo CAnbus at slcan0 if you run ifconfig.

Build the latest version using

    colcon build --packages-select onsmc_servo

Then open a new terminal and run, from ~/ros2_ws:

    source install/setup.bash
    ros2 launch onsmc_servo experiment.launch.py

Then, in another window, source ROS again and run

    ros2 topic pub --once /experiment/go std_msgs/msg/Bool "{data: 1}"

to start the experiment.

Note: I also experimented with queue length, and I got better performance with a short queue length (based on how I get commands and read them...could likely be doen smarter)

    sudo ip link set slcan0 txqueuelen 1

Worth experimenting with different queue lengths.

My default internal PI gains for the current loop are currently (20, 3) from current-loop-tracking experiments. The servo also seems (?) to have a deadzone of about 0.3 A...still working this out. Some of my SMC experiments have been able to get around this with sufficiently high gains (but then starts to look like bang-bang control). It's been tough tuning the servo itself, and hence my controller has also been tough to tune onto the servo.