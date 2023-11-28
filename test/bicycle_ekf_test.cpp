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

#include "pga_ekf/bicycle_ekf.h"
#include <gtest/gtest.h>
#include <array>
using namespace pga_ekf::bicycle;

// Test different Ctor calls and checks if states and uncertainties are similar
TEST(BicycleEKF_BasicTest, CtorTest)
{
    constexpr double initStd = 0.123;
    constexpr double initVariance = initStd * initStd;

    BicycleEKF<> ekf;
}

