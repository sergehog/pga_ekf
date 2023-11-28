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

// Test different Ctor calls and checks if states and uncertainties are similar
TEST(PgaEKF_BasicTest, CtorTest)
{
    constexpr double initStd = 0.123;
    constexpr double initVariance = initStd * initStd;

    PgaEKF::Enu enu{1, 2, 3, initStd, initStd, initStd};
    PgaEKF ekfEnu(enu);  // FIRST EKF

    PgaEKF::StateVector state;
    state << 1, -1. / 2, -2. / 2, -3. / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

    // variance scaled in a proper manner
    PgaEKF ekfState1(state, initVariance / 4.0);  // SECOND EKF

    auto uncertaintyMatrix = PgaEKF::UncertaintyMatrix::Identity() * initVariance / 4.0;
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
    }
}

// Checks if Ctor with ENU value is adequate
TEST(PgaEKF_BasicTest, EnuInitialization)
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

    const auto state = ekf.state();
    for (size_t i = kR01; i < kStateSize; i++)
    {
        // Velocities and Accelerations are zero
        EXPECT_EQ(state[i], 0.0);
    }

    // Orientation is Identity
    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0.0);

    // orientation uncertainty is 1 (maximum)
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.w(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.x(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.y(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1.0);
}

//! when speed=0, acc=0, position does not change, but uncertainty grows
TEST(PgaEKF_BasicTest, BasicPredict)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    const double processNoiseStd = .321;
    ekf.predict(0.1, processNoiseStd);

    // Motor is still normalized
    EXPECT_EQ(ekf.motorNorm(), 1.0);

    // Initial ENU values has not changed, as not velocity / acc
    EXPECT_EQ(ekf.filteredPosition().x, enu.x);
    EXPECT_EQ(ekf.filteredPosition().y, enu.y);
    EXPECT_EQ(ekf.filteredPosition().z, enu.z);

    // uncertainty grows approximately as sum of variances (not fully correct check though)
    EXPECT_NEAR(ekf.filteredPosition().stdX, sqrt(enu.stdX * enu.stdX + 4 * processNoiseStd * processNoiseStd), 1e-3);
    EXPECT_NEAR(ekf.filteredPosition().stdY, sqrt(enu.stdY * enu.stdY + 4 * processNoiseStd * processNoiseStd), 1e-3);
    EXPECT_NEAR(ekf.filteredPosition().stdZ, sqrt(enu.stdZ * enu.stdZ + 4 * processNoiseStd * processNoiseStd), 1e-3);

    // Orientation has not changed
    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0.0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0.0);

    // orientation uncertainty has grown
    EXPECT_GT(ekf.filteredOrientation().uncertainty.w(), 1.0);
    EXPECT_GT(ekf.filteredOrientation().uncertainty.x(), 1.0);
    EXPECT_GT(ekf.filteredOrientation().uncertainty.y(), 1.0);
    EXPECT_GT(ekf.filteredOrientation().uncertainty.z(), 1.0);
}

//! Checks if PgaEKF::updateImu works at all
TEST(PgaEKF_BasicTest, BasicUpdateImu)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    PgaEKF::Imu imu{0, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    ekf.updateImu(imu);

    // Motor normalized
    EXPECT_EQ(ekf.motorNorm(), 1.0);

    // After one IMU update orientation uncertainty decreased, everything else stay the same
    EXPECT_EQ(ekf.filteredPosition().x, enu.x);
    EXPECT_EQ(ekf.filteredPosition().y, enu.y);
    EXPECT_EQ(ekf.filteredPosition().z, enu.z);
    EXPECT_EQ(ekf.filteredPosition().stdX, enu.stdX);
    EXPECT_EQ(ekf.filteredPosition().stdY, enu.stdY);
    EXPECT_EQ(ekf.filteredPosition().stdZ, enu.stdZ);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    EXPECT_LT(ekf.filteredOrientation().uncertainty.w(), .5);
    EXPECT_LT(ekf.filteredOrientation().uncertainty.x(), .003);
    EXPECT_LT(ekf.filteredOrientation().uncertainty.y(), .003);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1);  // rotation around z is as uncertain as before
}

//! Checks if PgaEKF::updateEnu works at all
TEST(PgaEKF_BasicTest, BasicUpdateEnu)
{
    PgaEKF::Enu initEnu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(initEnu);

    PgaEKF::Enu updateEnu{6, 5, 4, 3, 2, 1};
    ekf.updateEnu(updateEnu);

    // Motor normalized
    EXPECT_EQ(ekf.motorNorm(), 1.0);

    const auto result = ekf.filteredPosition();

    // resulting ENU is somewhere in between
    EXPECT_LT(initEnu.x, result.x);
    EXPECT_LT(result.x, updateEnu.x);
    EXPECT_LT(initEnu.y, result.y);
    EXPECT_LT(result.y, updateEnu.y);
    EXPECT_LT(initEnu.z, result.z);
    EXPECT_LT(result.z, updateEnu.z);

    // resulting ENU incertainty is decreased
    EXPECT_LT(result.stdX, initEnu.stdX);
    EXPECT_LT(result.stdY, initEnu.stdY);
    EXPECT_LT(result.stdZ, initEnu.stdZ);

    // Orientation has not changed
    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);
}

// when orientation is uncertain -> ENU uncertainty grows with presence of acceleration
// TEST(PgaEKF_BasicTest, UncertaintyGrowsDuringAcceleration)
//{
//    constexpr double kLowUncertaintyStd = 1e-8;
//    constexpr double kLowUncertaintyVar = kLowUncertaintyStd * kLowUncertaintyStd;
//    PgaEKF::Enu enu{0, 0, 0, kLowUncertaintyStd, kLowUncertaintyStd, kLowUncertaintyStd};
//    // We have good assurance about ENU, but bad one about orientation
//    PgaEKF ekf(enu);
//
//    // Imu measurements are also accurate
//    PgaEKF::Imu imu{1, 0, -pga_ekf::kGravity, 0, 0, 0};
//    imu.stdAx = kLowUncertaintyStd;
//    imu.stdAy = kLowUncertaintyStd;
//    imu.stdAz = kLowUncertaintyStd;
//    imu.stdGx = kLowUncertaintyStd;
//    imu.stdGy = kLowUncertaintyStd;
//    imu.stdGz = kLowUncertaintyStd;
//
//    // First IMU update
//    ekf.updateImu(imu);
//
//    // Motor normalized
//    EXPECT_EQ(ekf.motorNorm(), 1.0);
//
//    // Before calling predict(), velocities are known to be 0
//    for(size_t i = kR01; i <= kR23; i++)
//    {
//        EXPECT_EQ(ekf.state()[i], 0.0);
//        EXPECT_EQ(ekf.uncertainty().row(i)[i], 0.0);
//    }
//    // this time acceleration is not affected, instead, orientation is "adjusted".
//    // ToDo: But why? Shouldn't X acceleration be already non-zero //EXPECT_NE(ekf.state()[kA01], 0.0);
//    EXPECT_EQ(ekf.state()[kA01], 0.0);
//
//    // Y,  Z acc are still zeros
//    EXPECT_EQ(ekf.state()[kA02], 0.0);
//    EXPECT_EQ(ekf.state()[kA03], 0.0);
//
//    std::cout << "State after Imu Update: " << ekf.state().transpose() << std::endl;
//    std::cout << "Variance after Imu Update: " << ekf.uncertainty().diagonal().transpose() << std::endl;
//    constexpr double kProcessNoiseStd = 0.01;
//    constexpr double kProcessNoiseVar = kProcessNoiseStd*kProcessNoiseStd;
//    ekf.predict(0.1, 0.01);
//    // Motor normalized
//    EXPECT_EQ(ekf.motorNorm(), 1.0);
//
//
//    std::cout << "State after Predict: " << ekf.state().transpose() << std::endl;
//    std::cout << "Variance after Predict: " << ekf.uncertainty().diagonal().transpose() << std::endl;
//
//    // Acceleration Uncertainty Grows During Predict
//    for(size_t i=kM01; i<kStateSize; i++)
//    {
//        if(i != kM12 && i != kM0123)
//        {
//            EXPECT_NEAR(ekf.uncertainty().row(i)[i], kProcessNoiseVar, 1e-8) << "i=" << i;
//        }
//    }
//
//    ekf.updateImu(imu);
//    // Motor normalized
//    EXPECT_EQ(ekf.motorNorm(), 1.0);
//
//    std::cout << "State after second Imu Update: " << ekf.state().transpose() << std::endl;
//    std::cout << "Variance after second Imu Update: " << ekf.uncertainty().diagonal().transpose() << std::endl;
//
//    // now acceleration shall get affected
//    // EXPECT_GT(ekf.state()[14], 0.0);
//    // while speed is still 0
//    EXPECT_EQ(ekf.state()[8], 0.0);
////    // Angular Momentum uncertainty stays zero all the time, as no gyro was involved
////    EXPECT_NEAR(ekf.uncertainty().row(11)[11], 0., 1e-7);
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