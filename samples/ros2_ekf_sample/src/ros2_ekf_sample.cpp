/*
 * This file is part of the PGA-EKF distribution (https://github.com/sergehog/pga_ekf)
 * Copyright (c) 2022 Sergey Smirnov / Seregium Oy.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nav_msgs/msg/odometry.hpp>
#include <pga_ekf/pga_ekf.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <memory>

using namespace std::chrono_literals;
using namespace pga_ekf;
using std::placeholders::_1;

class Ros2EkfSample : public rclcpp::Node
{
  public:
    Ros2EkfSample() : Node("ros2_ekf_sample")
    {
        _imu_subscription =
            this->create_subscription<sensor_msgs::msg::Imu>("imu", 2, std::bind(&Ros2EkfSample::_imuCallback, this, _1));
        _odom_publisher = this->create_publisher<nav_msgs::msg::Odometry>("odom", 2);
        declare_parameter<std::vector<double>>("initial_enu", {0.0, 0.0, 0.0});
        declare_parameter<double>("process_noise", 0.01);
        declare_parameter<double>("publish_fps", 5.0);
        auto enu_vec = get_parameter("initial_enu").as_double_array();
        _process_noise = get_parameter("process_noise").as_double();
        const auto publish_fps = std::max(0.1, get_parameter("publish_fps").as_double());
        _odom_publish_delay = 1.0 / publish_fps;
        assert(enu_vec.size() >= 3);
        PgaEKF::Enu enu{enu_vec[0], enu_vec[1], enu_vec[2]};
        _ekf = std::make_unique<PgaEKF>(enu);
    }

  private:
    void _imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        double current_time = double(msg->header.stamp.sec) + double(msg->header.stamp.nanosec) * 1e-9;
        double delta_time{};
        if (_previous_time > 0.0)
        {
            delta_time = current_time - _previous_time;
            _ekf->predict(delta_time, _process_noise);
            _previous_time = current_time;
        }

        PgaEKF::Imu imu{msg->linear_acceleration.x,
                        msg->linear_acceleration.y,
                        msg->linear_acceleration.z,
                        msg->angular_velocity.x,
                        msg->angular_velocity.y,
                        msg->angular_velocity.z,
                        msg->linear_acceleration_covariance[0],
                        msg->linear_acceleration_covariance[4],
                        msg->linear_acceleration_covariance[8],
                        msg->angular_velocity_covariance[0],
                        msg->angular_velocity_covariance[4],
                        msg->angular_velocity_covariance[8]};
        _ekf->updateImu(imu);

        if (current_time >= _previous_odom_publish_time + _odom_publish_delay)
        {
            nav_msgs::msg::Odometry odom{};
            odom.header.stamp = msg->header.stamp;
            odom.header.frame_id = "odom";
            odom.child_frame_id = msg->header.frame_id;
            auto enu = _ekf->filteredPosition();
            odom.pose.pose.position.x = enu.x;
            odom.pose.pose.position.y = enu.y;
            odom.pose.pose.position.z = enu.z;
            auto ori = _ekf->filteredOrientation();
            odom.pose.pose.orientation.w = ori.orientation.coeffs()[0];
            odom.pose.pose.orientation.x = ori.orientation.coeffs()[1];
            odom.pose.pose.orientation.z = ori.orientation.coeffs()[2];
            odom.pose.pose.orientation.w = ori.orientation.coeffs()[3];
            _previous_odom_publish_time = current_time;
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr _imu_subscription;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr _odom_publisher;
    std::unique_ptr<PgaEKF> _ekf{};
    double _previous_time = 0.0;
    double _process_noise{};
    double _previous_odom_publish_time = 0.0;
    double _odom_publish_delay{};
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Ros2EkfSample>());
    rclcpp::shutdown();
    return 0;
}