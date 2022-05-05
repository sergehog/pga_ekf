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


//! Tests PgaEKF.updateImu(..) function
//! @param tuple[0] - Initial (input) state
//! @param tuple[1] - Input IMU values
//! @param tuple[2] - Expected (output) state
class PgaEKF_UpdateImuTest
    : public ::testing::TestWithParam<
          std::tuple<std::array<double, PgaEKF::kStateSize>, PgaEKF::Imu, std::array<double, PgaEKF::kStateSize>>>
{
  public:
    constexpr static double kAccuracy = 1e-10;

    constexpr static std::array<double, PgaEKF::kStateSize> kStationaryIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kStationaryExpected = kStationaryIn;

    constexpr static PgaEKF::Imu kStationaryImu = {0, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    constexpr static PgaEKF::Imu kUpsideDownImu = {0, 0, pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

    //!
    std::array<double, PgaEKF::kStateSize> _expectedState{};
    PgaEKF::Imu _inputImu{};
    PgaEKF::StateVector _inputState;

    PgaEKF_UpdateImuTest()
    {
        std::array<double, PgaEKF::kStateSize> input{};
        std::tie(input, _inputImu, _expectedState) = GetParam();
        // convert array<...> into actual State vector
        for (std::size_t i = 0UL; i < PgaEKF::kStateSize; i++)
        {
            _inputState[i] = input[i];
        }
    }
};

TEST_P(PgaEKF_UpdateImuTest, ParametrizedTest)
{
    PgaEKF::UncertaintyMatrix uncertainty = PgaEKF::UncertaintyMatrix::Identity() * kAccuracy;

    PgaEKF ekf(_inputState, uncertainty);
    ekf.updateImu(_inputImu);

    for (std::size_t i = 0UL; i < PgaEKF::kStateSize; i++)
    {
        EXPECT_NEAR(ekf.state()[i], _expectedState[i], kAccuracy) << ", i=" << i;
    }

    //    for (std::size_t i = 0UL; i < PgaEKF::kStateSize; i++)
    //    {
    //        EXPECT_NEAR(ekf.uncertainty().row(i)[i], kProcessNoise + kAccuracy, kAccuracy) << "i=" << i;
    //    }
}

INSTANTIATE_TEST_CASE_P(InstantiationName,
                        PgaEKF_UpdateImuTest,
                        testing::Values(std::make_tuple(PgaEKF_UpdateImuTest::kStationaryIn,
                                                        PgaEKF_UpdateImuTest::kStationaryImu,
                                                        PgaEKF_UpdateImuTest::kStationaryExpected)
                                            //,
                                        //! TODO: this test instance suppose to fail, actually (o_O)
//                                        std::make_tuple(PgaEKF_UpdateImuTest::kOriginIn,
//                                                        PgaEKF_UpdateImuTest::kUpsideDownImu,
//                                                        PgaEKF_UpdateImuTest::kOriginExpected)
                                            ));
