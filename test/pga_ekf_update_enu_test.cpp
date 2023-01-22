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

//! Tests PgaEKF.updateEnu(..) function
//! @param tuple[0] - Initial (input) state
//! @param tuple[1] - Input ENU values
//! @param tuple[2] - Expected (output) state
class PgaEKF_UpdateEnuTest
    : public ::testing::TestWithParam<std::tuple<std::array<double, kStateSize>, PgaEKF::Enu, std::array<double, kStateSize>>>
{
  public:
    constexpr static double kHighAccuracy = 1e-10;
    constexpr static double kMediumAccuracy = 0.1;
    constexpr static double kBadAccuracy = 10.;

    constexpr static std::array<double, kStateSize> kStateOrigin = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kState123 = {1, -1./2, -2./2, -3./2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kStateX11 = {1, -11./2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kStateY17 = {1, 0, -17./2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, kStateSize> kStateZ27 = {1, 0, 0, -27./2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    constexpr static PgaEKF::Enu kOriginEnu = {0, 0, 0};
    constexpr static PgaEKF::Enu k123Enu = {1, 2, 3};
    constexpr static PgaEKF::Enu kX11Enu = {11, 0, 0};
    constexpr static PgaEKF::Enu kY17Enu = {0, 17, 0};
    constexpr static PgaEKF::Enu kZ27Enu = {0, 0, 27};


    std::array<double, kStateSize> _expectedState{};
    PgaEKF::Enu _inputEnu{};
    PgaEKF::StateVector _inputState;

    PgaEKF_UpdateEnuTest()
    {
        std::array<double, kStateSize> input{};
        std::tie(input, _inputEnu, _expectedState) = GetParam();
        // convert array<...> into actual State vector
        for (std::size_t i = 0UL; i < kStateSize; i++)
        {
            _inputState[i] = input[i];
        }
    }
};

//! Zero observation uncertainty -> State change to observed one
TEST_P(PgaEKF_UpdateEnuTest, PerfectObservationsTest)
{
    // Initial state uncertainty is pretty low, but we don't care
    PgaEKF::UncertaintyMatrix uncertainty = PgaEKF::UncertaintyMatrix::Identity() * kHighAccuracy;

    PgaEKF ekf(_inputState, uncertainty);

    // Observation uncertainty is zero
    _inputEnu.stdX = 0.0;
    _inputEnu.stdY = 0.0;
    _inputEnu.stdZ = 0.0;
    ekf.updateEnu(_inputEnu);

    for (std::size_t i = 0UL; i < kStateSize; i++)
    {
        EXPECT_NEAR(ekf.state()[i], _expectedState[i], kHighAccuracy) << ", i=" << i;
    }

    for (std::size_t i = 0UL; i < kStateSize; i++)
    {
        EXPECT_GE(ekf.uncertainty().row(i)[i], 0.0) << "i=" << i;
        EXPECT_LE(ekf.uncertainty().row(i)[i], kHighAccuracy) << "i=" << i;
    }
}

//! Observations have some inaccuracy -> State will be "weighted average" between estimated and observed
TEST_P(PgaEKF_UpdateEnuTest, UncertainObservationsTest)
{
    PgaEKF::UncertaintyMatrix uncertainty = PgaEKF::UncertaintyMatrix::Identity() * kHighAccuracy;

    PgaEKF ekf(_inputState, uncertainty);
    _inputEnu.stdX = kMediumAccuracy;
    _inputEnu.stdY = kMediumAccuracy;
    _inputEnu.stdZ = kMediumAccuracy;

    ekf.updateEnu(_inputEnu);

    // "weighted average" in 3D means that fused point is on the line between initial and given points
    const double x_direction = _expectedState[kM01] - _inputState[kM01];
    const double y_direction = _expectedState[kM02] - _inputState[kM02];
    const double z_direction = _expectedState[kM03] - _inputState[kM03];

    const double x_change = ekf.state()[kM01] - _inputState[kM01];
    const double y_change = ekf.state()[kM02] - _inputState[kM02];
    const double z_change = ekf.state()[kM03] - _inputState[kM03];

    const double x_change_rel = abs(x_direction) > 0.0 ? x_change/x_direction : 0.0;
    const double y_change_rel = abs(y_direction) > 0.0 ? y_change/y_direction : 0.0;
    const double z_change_rel = abs(z_direction) > 0.0 ? z_change/z_direction : 0.0;

    // also directions of relative change is in the 0..1 range
    EXPECT_LE(0, x_change_rel); EXPECT_LE(x_change_rel, 1);
    EXPECT_LE(0, y_change_rel); EXPECT_LE(y_change_rel, 1);
    EXPECT_LE(0, z_change_rel); EXPECT_LE(z_change_rel, 1);

    // check that actual state is on the line between _inputState and _expectedState
    if(x_change_rel > 0 && y_change_rel > 0)
    {
        EXPECT_NEAR(x_change_rel, y_change_rel, kHighAccuracy)
            << "[X: " << _inputState[kM01] << " <=> " << ekf.state()[kM01] << " <=> " << _expectedState[kM01] << "]"
            << "[Y: " << _inputState[kM03] << " <=> " << ekf.state()[kM03] << " <=> " << _expectedState[kM03] << "]";
    }

    if(x_change_rel > 0 && z_change_rel > 0)
    {
        EXPECT_NEAR(x_change_rel, z_change_rel, kHighAccuracy)
            << "[X: " << _inputState[kM01] << " <=> " << ekf.state()[kM01] << " <=> " << _expectedState[kM01] << "]"
            << "[Z: " << _inputState[kM02] << " <=> " << ekf.state()[kM02] << " <=> " << _expectedState[kM02] << "]";
    }

    if(y_change_rel > 0 && z_change_rel > 0)
    {
        EXPECT_NEAR(y_change_rel, z_change_rel, kHighAccuracy)
            << "[Y: " << _inputState[kM03] << " <=> " << ekf.state()[kM03] << " <=> " << _expectedState[kM03] << "]"
            << "[Z: " << _inputState[kM02] << " <=> " << ekf.state()[kM02] << " <=> " << _expectedState[kM02] << "]";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Start_From_Origin,
    PgaEKF_UpdateEnuTest,
    testing::Values(  //ToDo: rewrite with testinig::Combine(..)
        std::make_tuple(PgaEKF_UpdateEnuTest::kStateOrigin, PgaEKF_UpdateEnuTest::kOriginEnu, PgaEKF_UpdateEnuTest::kStateOrigin),
        std::make_tuple(PgaEKF_UpdateEnuTest::kStateOrigin, PgaEKF_UpdateEnuTest::k123Enu, PgaEKF_UpdateEnuTest::kState123),
        std::make_tuple(PgaEKF_UpdateEnuTest::kStateOrigin, PgaEKF_UpdateEnuTest::kX11Enu, PgaEKF_UpdateEnuTest::kStateX11),
        std::make_tuple(PgaEKF_UpdateEnuTest::kStateOrigin, PgaEKF_UpdateEnuTest::kY17Enu, PgaEKF_UpdateEnuTest::kStateY17),
        std::make_tuple(PgaEKF_UpdateEnuTest::kStateOrigin, PgaEKF_UpdateEnuTest::kZ27Enu, PgaEKF_UpdateEnuTest::kStateZ27)
        ));

INSTANTIATE_TEST_SUITE_P(
    Start_From_123,
    PgaEKF_UpdateEnuTest,
    testing::Values(  //ToDo: rewrite with testinig::Combine(..)
                std::make_tuple(PgaEKF_UpdateEnuTest::kState123, PgaEKF_UpdateEnuTest::kOriginEnu, PgaEKF_UpdateEnuTest::kStateOrigin),
                std::make_tuple(PgaEKF_UpdateEnuTest::kState123, PgaEKF_UpdateEnuTest::k123Enu, PgaEKF_UpdateEnuTest::kState123)//,
//                std::make_tuple(PgaEKF_UpdateEnuTest::kState123, PgaEKF_UpdateEnuTest::kX11Enu, PgaEKF_UpdateEnuTest::kStateX11),
//                std::make_tuple(PgaEKF_UpdateEnuTest::kState123, PgaEKF_UpdateEnuTest::kY17Enu, PgaEKF_UpdateEnuTest::kStateY17),
//                std::make_tuple(PgaEKF_UpdateEnuTest::kState123, PgaEKF_UpdateEnuTest::kZ27Enu, PgaEKF_UpdateEnuTest::kStateZ27)
        ));
