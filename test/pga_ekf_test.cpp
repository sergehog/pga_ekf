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

TEST(PgaEKF_BasicTest, CtorTest)
{
    constexpr double initStd = 0.123;
    constexpr double initVariance = initStd * initStd / 4.0;

    PgaEKF::Enu enu{1, 2, 3, initStd, initStd, initStd};
    PgaEKF ekfEnu(enu);  // FIRST EKF

    PgaEKF::StateVector state;
    state << 1, -1. / 2, -2. / 2, -3. / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    PgaEKF ekfState1(state, initVariance);  // SECOND EKF

    auto uncertaintyMatrix = PgaEKF::UncertaintyMatrix::Identity() * initVariance;
    PgaEKF ekfState2(state, uncertaintyMatrix);  // THIRD EKF

    // Check that EKF states are the same
    for (auto i = 0UL; i < kStateSize; i++)
    {
        EXPECT_EQ(ekfEnu.state()[i], ekfState1.state()[i]) << "i=" << i;
        EXPECT_EQ(ekfState1.state()[i], ekfState2.state()[i]) << "i=" << i;
        // in ekfState1 and ekfState2 uncertainties must also be the same
        EXPECT_EQ(ekfState1.uncertainty().row(i)[i], ekfState2.uncertainty().row(i)[i]) << "i=" << i;
    }

    // ekfEnu and ekfState1 must share same uncertainty only for XYZ components
    for (auto i = 1UL; i < 4; i++)
    {
        EXPECT_EQ(ekfEnu.uncertainty().row(i)[i], ekfState1.uncertainty().row(i)[i]) << "i=" << i;
        ;
    }
}

TEST(PgaEKF_BasicTest, EnuInitializationTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);

    EXPECT_EQ(ekf.motorNorm(), 1.0);

    // Check output values
    EXPECT_EQ(ekf.filteredPosition().x, enu.x);
    EXPECT_EQ(ekf.filteredPosition().y, enu.y);
    EXPECT_EQ(ekf.filteredPosition().z, enu.z);
    EXPECT_EQ(ekf.filteredPosition().stdX, enu.stdX);
    EXPECT_EQ(ekf.filteredPosition().stdY, enu.stdY);
    EXPECT_EQ(ekf.filteredPosition().stdZ, enu.stdZ);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0.0);

    // Velocities are zero
    const auto state = ekf.state();
    EXPECT_EQ(state[kR01], 0.0);
    EXPECT_EQ(state[kR02], 0.0);
    EXPECT_EQ(state[kR03], 0.0);
    EXPECT_EQ(state[kR12], 0.0);
    EXPECT_EQ(state[kR13], 0.0);
    EXPECT_EQ(state[kR23], 0.0);
    // Accelerations are zero
    EXPECT_EQ(state[kA01], 0.0);
    EXPECT_EQ(state[kA02], 0.0);
    EXPECT_EQ(state[kA03], 0.0);

    // orientation is unknown -> uncertainty == 1
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.w(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.x(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.y(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1.0);
}

//! when speed=0, acc=0, position does not change, but uncertainty grows
TEST(PgaEKF_BasicTest, BasicPredictTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    const double processNoise = .321;
    ekf.predict(0.1, processNoise);
    // Check default values
    EXPECT_EQ(ekf.filteredPosition().x, enu.x);
    EXPECT_EQ(ekf.filteredPosition().y, enu.y);
    EXPECT_EQ(ekf.filteredPosition().z, enu.z);

    EXPECT_NEAR(ekf.filteredPosition().stdX, enu.stdX, sqrt(processNoise));
    EXPECT_NEAR(ekf.filteredPosition().stdY, enu.stdY, sqrt(processNoise));
    EXPECT_NEAR(ekf.filteredPosition().stdZ, enu.stdZ, sqrt(processNoise));

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0.0);

    // orientaiton is unknown -> uncertainty == 1
    //    EXPECT_EQ(ekf.filteredOrientation().uncertainty.w(), 1 + processNoise);
    //    EXPECT_EQ(ekf.filteredOrientation().uncertainty.x(), 1 + processNoise);
    //    EXPECT_EQ(ekf.filteredOrientation().uncertainty.y(), 1 + processNoise);
    //    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1 + processNoise);
}

//! Checks that PgaEKF::updateImu works at all
TEST(PgaEKF_BasicTest, BasicUpdateImuTest)
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

TEST(PgaEKF_BasicTest, UncertaintyGrowDuringAccelerationTest)
{
    PgaEKF::Enu enu{0, 0, 0, 1e-8, 1e-8, 1e-8};
    PgaEKF ekf(enu);
    std::cout << "State before Imu Update: " << ekf.state().transpose() << std::endl;
    PgaEKF::Imu imu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10};
    ekf.updateImu(imu);
    // velocity and acceleration are known to be 0 with 0 uncertainty -> it stays this way
    EXPECT_EQ(ekf.uncertainty().row(14)[14], 0.0);
    EXPECT_EQ(ekf.uncertainty().row(15)[15], 0.0);
    EXPECT_EQ(ekf.uncertainty().row(16)[16], 0.0);
    // this time acceleration is not affected, instead, orientation is adjusted
    EXPECT_EQ(ekf.state()[14], 0.0);
    EXPECT_GT(ekf.state()[5], 0.0);

    std::cout << "State after Imu Update: " << ekf.state().transpose() << std::endl;
    ekf.predict(0.1, 0.01);

    std::cout << "State after Predict: " << ekf.state().transpose() << std::endl;

    // Acceleration Uncertainty Grows During Predict
    EXPECT_GT(ekf.uncertainty().row(14)[14], 0.0);
    EXPECT_GT(ekf.uncertainty().row(15)[15], 0.0);
    EXPECT_GT(ekf.uncertainty().row(16)[16], 0.0);

    ekf.updateImu(imu);
    std::cout << "State after second Imu Update: " << ekf.state().transpose() << std::endl;

    // now acceleration shall get affected
    // EXPECT_GT(ekf.state()[14], 0.0);
    // while speed is still 0
    EXPECT_EQ(ekf.state()[8], 0.0);

    // Angular Momentum uncertainty stays zero all the time, as no gyro was involved
    EXPECT_NEAR(ekf.uncertainty().row(11)[11], 0., 1e-7);
    EXPECT_NEAR(ekf.uncertainty().row(12)[12], 0., 1e-7);
    EXPECT_NEAR(ekf.uncertainty().row(13)[13], 0., 1e-7);
}

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