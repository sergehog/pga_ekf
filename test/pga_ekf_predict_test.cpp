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

//! Tests PgaEKF.predict(..) function
//! @param tuple[0] - Initial (input) state
//! @param tuple[1] - Expected (output) state
class PgaEKF_PredictTest
    : public ::testing::TestWithParam<std::tuple<std::array<double, kStateSize>, std::array<double, kStateSize>>>
{
  public:
    constexpr static double kProcessNoise = 0.1;
    constexpr static double kVelocity = 1;
    constexpr static double kAccel = 1;
    constexpr static double kTimeDelta = 0.1;
    constexpr static double kAccuracy = 1e-10;

    constexpr static std::array<double, kStateSize> kStationaryIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kStationaryExpected = kStationaryIn;
    constexpr static std::array<double, kStateSize> kStationary2In = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kStationary2Expected = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kXVelocityIn = {1, 0, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kXVelocityExpected =
        {1, -kVelocity* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kYVelocityIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kYVelocityExpected =
        {1, 0, -kVelocity* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kZVelocityIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kZVelocityExpected =
        {1, 0, 0, -kVelocity* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kXAccelerationIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kAccel, 0, 0};
    constexpr static std::array<double, kStateSize> kXAccelerationExpected =
        {1, -kAccel* kTimeDelta* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kAccel* kTimeDelta, 0, 0, 0, 0, 0, kAccel, 0, 0};
    constexpr static std::array<double, kStateSize> kYAccelerationIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kAccel, 0};
    constexpr static std::array<double, kStateSize> kYAccelerationExpected =
        {1, 0, -kAccel* kTimeDelta* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kAccel* kTimeDelta, 0, 0, 0, 0, 0, kAccel, 0};
    constexpr static std::array<double, kStateSize> kZAccelerationIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kAccel};
    constexpr static std::array<double, kStateSize> kZAccelerationExpected =
        {1, 0, 0, -kAccel* kTimeDelta* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kAccel* kTimeDelta, 0, 0, 0, 0, 0, kAccel};

  public:
    std::array<double, kStateSize> _expectedState{};
    PgaEKF::StateVector _inputState;

    PgaEKF_PredictTest()
    {
        std::array<double, kStateSize> input{};
        std::tie(input, _expectedState) = GetParam();

        // convert array<...> into actual State vector
        for (std::size_t i = 0UL; i < kStateSize; i++)
        {
            _inputState[i] = input[i];
        }
    }
};

TEST_P(PgaEKF_PredictTest, ParametrizedTest)
{
    PgaEKF::UncertaintyMatrix uncertainty = PgaEKF::UncertaintyMatrix::Identity() * kAccuracy;

    PgaEKF ekf(_inputState, uncertainty);
    ekf.predict(kTimeDelta, kProcessNoise);

    for (std::size_t i = 0UL; i < kStateSize; i++)
    {
        EXPECT_NEAR(ekf.state()[i], _expectedState[i], kAccuracy) << ", i=" << i;
    }

    for (std::size_t i = 0UL; i < kStateSize; i++)
    {
        EXPECT_NEAR(ekf.uncertainty().row(i)[i], kProcessNoise + kAccuracy, kAccuracy) << "i=" << i;
    }
}

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    PgaEKF_PredictTest,
    testing::Values(std::make_tuple(PgaEKF_PredictTest::kStationaryIn, PgaEKF_PredictTest::kStationaryExpected),
                    std::make_tuple(PgaEKF_PredictTest::kStationary2In, PgaEKF_PredictTest::kStationary2Expected),
                    std::make_tuple(PgaEKF_PredictTest::kXVelocityIn, PgaEKF_PredictTest::kXVelocityExpected),
                    std::make_tuple(PgaEKF_PredictTest::kYVelocityIn, PgaEKF_PredictTest::kYVelocityExpected),
                    std::make_tuple(PgaEKF_PredictTest::kZVelocityIn, PgaEKF_PredictTest::kZVelocityExpected),
                    std::make_tuple(PgaEKF_PredictTest::kXAccelerationIn, PgaEKF_PredictTest::kXAccelerationExpected),
                    std::make_tuple(PgaEKF_PredictTest::kYAccelerationIn, PgaEKF_PredictTest::kYAccelerationExpected),
                    std::make_tuple(PgaEKF_PredictTest::kZAccelerationIn, PgaEKF_PredictTest::kZAccelerationExpected)));
