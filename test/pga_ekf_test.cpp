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

#include <gtest/gtest.h>
#include <pga_ekf/pga_ekf.h>
#include <array>
using namespace pga_ekf;

TEST(PgaEKF_Test, BasicTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    // Check default values
    EXPECT_EQ(ekf.filteredPosition().x, 1);
    EXPECT_EQ(ekf.filteredPosition().y, 2);
    EXPECT_EQ(ekf.filteredPosition().z, 3);
    EXPECT_EQ(ekf.filteredPosition().stdX, 4);
    EXPECT_EQ(ekf.filteredPosition().stdY, 5);
    EXPECT_EQ(ekf.filteredPosition().stdZ, 6);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    // orientation is unknown -> uncertainty == 1
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.x(), 1);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.y(), 1);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1);
}

//! when speed=0, acc=0, position does not change, but uncertanty grows
TEST(PgaEKF_Test, BasicPredictTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    const double processNoise = .02;
    ekf.predict(0.1, processNoise);
    // Check default values
    EXPECT_EQ(ekf.filteredPosition().x, 1);
    EXPECT_EQ(ekf.filteredPosition().y, 2);
    EXPECT_EQ(ekf.filteredPosition().z, 3);
    EXPECT_EQ(ekf.filteredPosition().stdX, 4 + processNoise * 2);
    EXPECT_EQ(ekf.filteredPosition().stdY, 5 + processNoise * 2);
    EXPECT_EQ(ekf.filteredPosition().stdZ, 6 + processNoise * 2);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    // orientaiton is unknown -> uncertainty == 1
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.w(), 1 + processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.x(), 1 + processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.y(), 1 + processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1 + processNoise);
}

//! Checks that PgaEKF::updateImu works at all
TEST(PgaEKF_Test, BasicUpdateImuTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    PgaEKF::Imu imu{0, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    ekf.updateImu(imu);

    // After one IMU update orientation uncertainty decreased
    // everything else stay the same
    EXPECT_EQ(ekf.filteredPosition().x, 1);
    EXPECT_EQ(ekf.filteredPosition().y, 2);
    EXPECT_EQ(ekf.filteredPosition().z, 3);
    EXPECT_EQ(ekf.filteredPosition().stdX, 4);
    EXPECT_EQ(ekf.filteredPosition().stdY, 5);
    EXPECT_EQ(ekf.filteredPosition().stdZ, 6);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    EXPECT_LT(ekf.filteredOrientation().uncertainty.w(), .5);
    EXPECT_LT(ekf.filteredOrientation().uncertainty.x(), .003);
    EXPECT_LT(ekf.filteredOrientation().uncertainty.y(), .003);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1);
}


// TEST(PgaEKFTest, AccPositiveXTest)
//{
//     PgaEKF::Enu enu{0, 0, 0, 0.001, 0.001, 0.001};
//     PgaEKF ekf(enu);
//     std::cout << "Initial State: " << ekf.state().transpose() << std::endl;
//     std::cout << "Initial Uncertainty: " << std::endl;
//     std::cout << ekf.uncertainty() << std::endl << std::endl;
//     //    EXPECT_NEAR(ekf.uncertainty().row(5)[5], 1, 1e-10);
//     //    EXPECT_NEAR(ekf.uncertainty().row(6)[6], 1, 1e-10);
//     //    EXPECT_NEAR(ekf.uncertainty().row(7)[7], 1, 1e-10);
//     PgaEKF::Imu imu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
//     ekf.updateImu(imu);
//     //    EXPECT_NEAR(ekf.uncertainty().row(5)[5], 1, 1e-10);
//     //    EXPECT_LT(ekf.uncertainty().row(6)[6], 0.003);
//     //    EXPECT_LT(ekf.uncertainty().row(7)[7], 0.003);
//
//     std::cout << "State after IMU update: " << ekf.state().transpose() << std::endl;
//     std::cout << "Uncertainty after IMU update: " << std::endl;
//     std::cout << ekf.uncertainty() << std::endl << std::endl;
// }

// TEST(PgaEKFTest, MovingXPositiveTest)
//{
//     PgaEKF::Enu enu{0, 0, 0, 0.001, 0.001, 0.001};
//     PgaEKF ekf(enu);
//
//     std::cout << "Initial State: " << ekf.state().transpose() << std::endl;
//     PgaEKF::Imu _inputImu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
//     ekf.updateImu(_inputImu);
//     std::cout << "State after Imu Update: " << ekf.state().transpose() << std::endl;
// }

// TEST(PgaEKFTest, UncertaintyGrowDuringAccelerationTest)
//{
//     PgaEKF::Enu enu{0, 0, 0, 1e-8, 1e-8, 1e-8};
//     PgaEKF ekf(enu);
//     std::cout << "State before Imu Update: " << ekf.state().transpose() << std::endl;
//     PgaEKF::Imu _inputImu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10};
//     ekf.updateImu(_inputImu);
//     std::cout << "State after Imu Update: " << ekf.state().transpose() << std::endl;
//
//     ekf.predict(0.1, 1e-8);
//
//     std::cout << "State after Predict: " << ekf.state().transpose() << std::endl;
//
//     // Orientation Uncertainty Grows During Acceleration
//     EXPECT_GE(ekf.uncertainty().row(8)[8], 0.01);
//     EXPECT_GE(ekf.uncertainty().row(9)[9], 0.01);
//     EXPECT_GE(ekf.uncertainty().row(10)[10], 0.01);
////    std::cout << "Uncertainty after predict: " << std::endl;
////    std::cout << ekf.uncertainty() << std::endl << std::endl;
//
//    // Angular Momentum uncertainty stays zero
////    EXPECT_NEAR(ekf.uncertainty().row(11)[11], 0., 1e-7);
////    EXPECT_NEAR(ekf.uncertainty().row(12)[12], 0., 1e-7);
////    EXPECT_NEAR(ekf.uncertainty().row(13)[13], 0., 1e-7);
//}

// TEST(PgaEKFTest, MoveTowardsPositiveXTest)
//{
//     PgaEKF::Enu enu{0, 0, 0, 0.001, 0.001, 0.001};
//     PgaEKF ekf(enu);
//
//     std::cout << "Initial State: " << ekf.state().transpose() << std::endl;
//
//     PgaEKF::Imu _inputImu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
//
//     for(int i=0; i<2; i++)
//     {
//         std::cout << "==================== IMU ===========================" << std::endl;
//         ekf.updateImu(_inputImu);
//         std::cout << "State after update: " << ekf.state().transpose() << std::endl;
//         std::cout << "Uncertainty after update: " << std::endl;
//         std::cout << ekf.uncertainty() << std::endl << std::endl;
//         ekf.predict(0.1);
//         std::cout << "State after predict: " << ekf.state().transpose() << std::endl;
//         std::cout << "Uncertainty after predict: " << std::endl;
//         std::cout << ekf.uncertainty() << std::endl << std::endl;
//     }
//
//     std::cout << "================== ENU =============================" << std::endl;
//     enu.x = 0.01;
//     //ekf.updateEnu(enu);
//
// }

//        std::cout << "Position: " << ekf.filteredEnu().x << "," << ekf.filteredEnu().y << "," << ekf.filteredPosition().z
//                  << std::endl;
//        std::cout << "Position uncertainty: " << ekf.filteredEnu().stdX << "," << ekf.filteredPosition().stdY << ","
//                  << ekf.filteredPosition().stdZ << std::endl;
//        std::cout << "Orientation: " << ekf.filteredOrientation().orientation.coeffs().transpose() << std::endl;
//        std::cout << "Orientation uncertainty: " << ekf.filteredOrientation().uncertainty.coeffs().transpose()
//                  << std::endl;