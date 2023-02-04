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

//! Tests PgaEKF.updateEnu(..) function
//! @param tuple[0] - Initialization ENU coordinate
//! @param tuple[1] - Input ENU, but also Expected (output) state

class PgaEKF_UpdateEnuTest : public ::testing::TestWithParam<std::tuple<PgaEKF::Enu, PgaEKF::Enu>>
{

  public:
    constexpr static double kHighAccuracy = 1e-10;
    constexpr static double kMediumAccuracy = 0.1;
    constexpr static double kBadAccuracy = 3.;

    constexpr static PgaEKF::Enu kOriginEnu = {0, 0, 0};
    constexpr static PgaEKF::Enu k123Enu = {1, 2, 3};
    constexpr static PgaEKF::Enu kX1Enu = {1, 0, 0};
    constexpr static PgaEKF::Enu kY2Enu = {0, 2, 0};
    constexpr static PgaEKF::Enu kZ3Enu = {0, 0, 3};

    PgaEKF::Enu _initializationEnu{};
    PgaEKF::Enu _inputAndExpectedEnu{};

    PgaEKF_UpdateEnuTest() { std::tie(_initializationEnu, _inputAndExpectedEnu) = GetParam(); }

    static void setEnuStd(PgaEKF::Enu& enu, double stdValue)
    {
        enu.stdX = stdValue;
        enu.stdY = stdValue;
        enu.stdZ = stdValue;
    }
};

//! Zero observation uncertainty -> State change to observed one
TEST_P(PgaEKF_UpdateEnuTest, PerfectObservationsTest)
{
    // Initial state uncertainty is pretty low
    setEnuStd(_initializationEnu, kBadAccuracy);

    PgaEKF ekf(_initializationEnu);

    // Observation uncertainty is zero
    setEnuStd(_inputAndExpectedEnu, kHighAccuracy);

    ekf.updateEnu(_inputAndExpectedEnu);

    auto outEnu = ekf.filteredPosition();
    EXPECT_NEAR(outEnu.x, _inputAndExpectedEnu.x, kHighAccuracy);
    EXPECT_NEAR(outEnu.y, _inputAndExpectedEnu.y, kHighAccuracy);
    EXPECT_NEAR(outEnu.z, _inputAndExpectedEnu.z, kHighAccuracy);

    // TEST: (0.0 <= uncertainty <= kHighAccuracy)
    EXPECT_LT(0.0, outEnu.stdX);
    EXPECT_LT(outEnu.stdX, kBadAccuracy);
    // EXPECT_LE(outEnu.stdX, 2*kHighAccuracy);

    EXPECT_LT(0.0, outEnu.stdY);
    EXPECT_LT(outEnu.stdY, kBadAccuracy);
    // EXPECT_LE(outEnu.stdY, 2*kHighAccuracy);

    EXPECT_LT(0.0, outEnu.stdZ);
    EXPECT_LT(outEnu.stdZ, kBadAccuracy);
    // EXPECT_LE(outEnu.stdZ, 2*kHighAccuracy);
}

//! Observations have some inaccuracy -> State will be "weighted average" between estimated and observed
TEST_P(PgaEKF_UpdateEnuTest, UncertainObservationsTest)
{

    setEnuStd(_initializationEnu, kBadAccuracy);
    PgaEKF ekf(_initializationEnu);

    // Observation uncertainty is zero
    setEnuStd(_inputAndExpectedEnu, kMediumAccuracy);

    ekf.updateEnu(_inputAndExpectedEnu);
    auto outEnu = ekf.filteredPosition();

    // "weighted average" in 3D means that fused point is on the line between initial and given points
    const double x_direction = _inputAndExpectedEnu.x - _initializationEnu.x;
    const double y_direction = _inputAndExpectedEnu.y - _initializationEnu.y;
    const double z_direction = _inputAndExpectedEnu.z - _initializationEnu.z;

    const double x_change = outEnu.x - _initializationEnu.x;
    const double y_change = outEnu.y - _initializationEnu.y;
    const double z_change = outEnu.z - _initializationEnu.z;

    const double x_change_rel = abs(x_direction) > 0.0 ? x_change / x_direction : 0.0;
    const double y_change_rel = abs(y_direction) > 0.0 ? y_change / y_direction : 0.0;
    const double z_change_rel = abs(z_direction) > 0.0 ? z_change / z_direction : 0.0;

    // also directions of relative change is in the 0..1 range
    EXPECT_LE(0, x_change_rel);
    EXPECT_LE(x_change_rel, 1);
    EXPECT_LE(0, y_change_rel);
    EXPECT_LE(y_change_rel, 1);
    EXPECT_LE(0, z_change_rel);
    EXPECT_LE(z_change_rel, 1);

    // check that actual state is on the line between _inputState and _expectedState
    if (x_change_rel > 0 && y_change_rel > 0)
    {
        EXPECT_NEAR(x_change_rel, y_change_rel, kHighAccuracy)
            << "[X: " << _initializationEnu.x << " <=> " << outEnu.x << " <=> " << _inputAndExpectedEnu.x << "]"
            << "[Y: " << _initializationEnu.y << " <=> " << outEnu.y << " <=> " << _inputAndExpectedEnu.y << "]";
    }

    if (x_change_rel > 0 && z_change_rel > 0)
    {
        EXPECT_NEAR(x_change_rel, z_change_rel, kHighAccuracy)
            << "[X: " << _initializationEnu.x << " <=> " << outEnu.x << " <=> " << _inputAndExpectedEnu.x << "]"
            << "[Z: " << _initializationEnu.z << " <=> " << outEnu.z << " <=> " << _inputAndExpectedEnu.z << "]";
    }

    if (y_change_rel > 0 && z_change_rel > 0)
    {
        EXPECT_NEAR(y_change_rel, z_change_rel, kHighAccuracy)
            << "[Y: " << _initializationEnu.y << " <=> " << outEnu.y << " <=> " << _inputAndExpectedEnu.z << "]"
            << "[Z: " << _initializationEnu.z << " <=> " << outEnu.z << " <=> " << _inputAndExpectedEnu.y << "]";
    }
}

INSTANTIATE_TEST_SUITE_P(Start_From_Origin,
                         PgaEKF_UpdateEnuTest,
                         testing::Values(  // ToDo: rewrite with testinig::Combine(..)
                             std::make_tuple(PgaEKF_UpdateEnuTest::kOriginEnu, PgaEKF_UpdateEnuTest::kOriginEnu),
                             std::make_tuple(PgaEKF_UpdateEnuTest::kOriginEnu, PgaEKF_UpdateEnuTest::k123Enu),
                             std::make_tuple(PgaEKF_UpdateEnuTest::kOriginEnu, PgaEKF_UpdateEnuTest::kX1Enu),
                             std::make_tuple(PgaEKF_UpdateEnuTest::kOriginEnu, PgaEKF_UpdateEnuTest::kY2Enu),
                             std::make_tuple(PgaEKF_UpdateEnuTest::kOriginEnu, PgaEKF_UpdateEnuTest::kZ3Enu)));

INSTANTIATE_TEST_SUITE_P(
    Start_From_123,
    PgaEKF_UpdateEnuTest,
    testing::Values(  // ToDo: rewrite with testinig::Combine(..)
        std::make_tuple(PgaEKF_UpdateEnuTest::k123Enu, PgaEKF_UpdateEnuTest::kOriginEnu),
        std::make_tuple(PgaEKF_UpdateEnuTest::k123Enu, PgaEKF_UpdateEnuTest::k123Enu),
        std::make_tuple(PgaEKF_UpdateEnuTest::kX1Enu, PgaEKF_UpdateEnuTest::kOriginEnu),
        std::make_tuple(PgaEKF_UpdateEnuTest::kY2Enu, PgaEKF_UpdateEnuTest::kOriginEnu),
        std::make_tuple(PgaEKF_UpdateEnuTest::kZ3Enu, PgaEKF_UpdateEnuTest::kOriginEnu)
        // std::make_tuple(PgaEKF_UpdateEnuTest::kState123, PgaEKF_UpdateEnuTest::kX1Enu, PgaEKF_UpdateEnuTest::kStateX1)

        ));
