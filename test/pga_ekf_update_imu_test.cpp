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

//! PgaEKF::StateVector based on Eigen::Matrix is not convenient to use in parametrized unit-tests,
//! so here we define more convenient array-based type
using StateVectorArray = std::array<double, kStateSize>;

//! Tests PgaEKF.updateImu(..) function
//! @param tuple[0] - Initial (input) state
//! @param tuple[1] - Input IMU values
//! @param tuple[2] - Expected (output) state
class PgaEKF_UpdateImuTest : public ::testing::TestWithParam<std::tuple<StateVectorArray, PgaEKF::Imu, StateVectorArray>>
{
  public:
    constexpr static double kInitialVariance = 0.01;

    constexpr static StateVectorArray kStateOrigin = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    constexpr static PgaEKF::Imu kStationaryImu = {0, 0, -pga_ekf::kGravity, 0, 0, 0};
    constexpr static PgaEKF::Imu kUpsideDownImu = {0, 0, pga_ekf::kGravity, 0, 0, 0};

    StateVectorArray _expectedState{};
    PgaEKF::Imu _inputImu{};
    PgaEKF::StateVector _inputState;

    PgaEKF_UpdateImuTest()
    {
        StateVectorArray input{};
        std::tie(input, _inputImu, _expectedState) = GetParam();
        // convert array<...> into actual State vector
        for (std::size_t i = 0UL; i < kStateSize; i++)
        {
            _inputState[i] = input[i];
        }
    }

    static void setImuStd(PgaEKF::Imu& imu, const double stdValue = 0.0)
    {
        imu.stdAx = stdValue;
        imu.stdAy = stdValue;
        imu.stdAz = stdValue;
        imu.stdGx = stdValue;
        imu.stdGy = stdValue;
        imu.stdGz = stdValue;
    }
};

//! Zero observation uncertainty -> State change as expected
TEST_P(PgaEKF_UpdateImuTest, PerfectObservationsTest)
{
    PgaEKF::UncertaintyMatrix uncertainty = PgaEKF::UncertaintyMatrix::Identity() * kInitialVariance;

    PgaEKF ekf(_inputState, uncertainty);

    setImuStd(_inputImu, 1e-10);  // std=0 => perfect observations

    ekf.updateImu(_inputImu);

    for (std::size_t i = 0UL; i < kStateSize; i++)
    {
        EXPECT_NEAR(ekf.state()[i], _expectedState[i], 1e-10) << ", i=" << i;
    }

    //    for (std::size_t i = 0UL; i < kStateSize; i++)
    //    {
    //        EXPECT_NEAR(ekf.uncertainty().row(i)[i], kUncertainty, kUncertainty) << "i=" << i;
    //    }
}

INSTANTIATE_TEST_SUITE_P(InstantiationName,
                         PgaEKF_UpdateImuTest,
                         testing::Values(std::make_tuple(PgaEKF_UpdateImuTest::kStateOrigin,
                                                         PgaEKF_UpdateImuTest::kStationaryImu,
                                                         PgaEKF_UpdateImuTest::kStateOrigin)
                                         // std::make_tuple(PgaEKF_UpdateImuTest::kStateOrigin,
                                         // PgaEKF_UpdateImuTest::kUpsideDownImu, PgaEKF_UpdateImuTest::kStateOrigin)
                                         ));

TEST(PgaEKF_UpdateTest, AccPositiveXTest)
{
    PgaEKF::Enu enu{0, 0, 0, 0.001, 0.001, 0.001};
    PgaEKF ekf(enu);
    std::cout << "Initial State: " << ekf.state().transpose() << std::endl;
    std::cout << "Initial Uncertainty: " << std::endl;
    std::cout << ekf.uncertainty() << std::endl << std::endl;
    EXPECT_NEAR(ekf.uncertainty().row(kMSC)[kMSC], 1, 1e-10);
    EXPECT_NEAR(ekf.uncertainty().row(kM12)[kM12], 1, 1e-10);
    EXPECT_NEAR(ekf.uncertainty().row(kM13)[kM13], 1, 1e-10);
    EXPECT_NEAR(ekf.uncertainty().row(kM23)[kM23], 1, 1e-10);
    PgaEKF::Imu imu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    ekf.updateImu(imu);
    EXPECT_LT(ekf.uncertainty().row(kMSC)[kMSC], 0.003);
    EXPECT_NEAR(ekf.uncertainty().row(kM12)[kM12], 1, 0.003);
    EXPECT_LT(ekf.uncertainty().row(kM13)[kM13], 0.003);
    EXPECT_LT(ekf.uncertainty().row(kM23)[kM23], 0.003);

    std::cout << "State after IMU update: " << ekf.state().transpose() << std::endl;
    std::cout << "Uncertainty after IMU update: " << std::endl;
    std::cout << ekf.uncertainty() << std::endl << std::endl;
}
