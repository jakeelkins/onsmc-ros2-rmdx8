from launch import LaunchDescription
import launch_ros.actions

def generate_launch_description():
    return LaunchDescription([
        launch_ros.actions.Node(
            namespace= "experiment", package='onsmc_servo', executable='time_publisher'),
        launch_ros.actions.Node(
            namespace= "experiment", package='onsmc_servo', executable='trajectory_generator'),
        launch_ros.actions.Node(
            namespace= "experiment", package='onsmc_servo', executable='controller'),
        launch_ros.actions.Node(
            namespace= "experiment", package='onsmc_servo', executable='reader'),
        launch_ros.actions.Node(
            namespace= "experiment", package='onsmc_servo', executable='commander'),
        launch_ros.actions.Node(
            namespace= "experiment", package='onsmc_servo', executable='writer'),
    ])