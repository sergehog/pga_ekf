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

#ifndef PGA_EKF_PGA_EKF_H
#define PGA_EKF_PGA_EKF_H

#include <Eigen/Dense>
#include <iostream>
#include <utility>

namespace pga_ekf
{

//! Named indexes of the State Vector
//! Intentionally made as a enum, such that visibility is inside of the namespace
enum StateIndex : std::size_t
{
    /// 8 components of the PGA Motor,
    /// Encodes Body Frame Pose in some global (ENU-like) coordinate frame
    kMSC = 0,
    kM01 = 1,
    kM02 = 2,
    kM03 = 3,
    kM12 = 4,
    kM13 = 5,
    kM23 = 6,
    kM0123 = 7,

    /// 6 components of the Rate Bivector
    /// Encodes Velocity (including Angular) in body frame
    kR01 = 8,
    kR02 = 9,
    kR03 = 10,
    kR12 = 11,
    kR13 = 12,
    kR23 = 13,

    /// 3 components of Acceleration Bivector
    /// if body is stationary, acceleration is zero
    kA01 = 14,
    kA02 = 15,
    kA03 = 16,

    kStateSize = 17
};

//! Gravity constant
constexpr double kGravity = 9.80665;

//! EKF filter for 2 types of inputs:
//! - IMU data in body-frame
//! - ENU Position in world-frame - can be obtained from GPS or some other sensor
//! We loosely call global coordinate frame "ENU",
//! however, only real constraint for it is that last "U" coordinate points up
class PgaEKF
{
  public:
    static constexpr std::size_t kImuSize = 6;
    static constexpr std::size_t kEnuSize = 3;
    using StateVector = Eigen::Matrix<double, kStateSize, 1>;
    using ImuVector = Eigen::Matrix<double, kImuSize, 1>;
    using EnuVector = Eigen::Matrix<double, kEnuSize, 1>;

    using UncertaintyMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;
    using ProcessNoiseMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;
    using ImuUncertainty = Eigen::Matrix<double, kImuSize, kImuSize>;
    using EnuUncertainty = Eigen::Matrix<double, kEnuSize, kEnuSize>;

    struct Imu
    {
        double ax, ay, az;  //!> accelerometer values
        double gx, gy, gz;  //!> gyroscope values

        double stdAx, stdAy, stdAz;  //!> standard deviation of accelerometer values
        double stdGx, stdGy, stdGz;  //!> standard deviation of gyroscope values

        [[nodiscard]] ImuVector vector() const
        {
            Eigen::Matrix<double, kImuSize, 1> vec;
            vec << ax, ay, az, gx, gy, gz;
            return vec;
        }

        [[nodiscard]] ImuUncertainty uncertainty() const
        {
            Eigen::DiagonalMatrix<double, kImuSize> uncertainty;
            // convert Standard Deviation into Variance
            uncertainty.diagonal() << stdAx * stdAx, stdAy * stdAy, stdAz * stdAz, stdGx * stdGx, stdGy * stdGy, stdGz * stdGz;
            return uncertainty.toDenseMatrix();
        }
    };

    //! Positive Z in ENU measurements must be true UP (i.e. opposite of gravity)
    //! X, Y not necessarily need to correspond to East-North
    struct Enu
    {
        double x, y, z;           //!> Global Position values
        double stdX, stdY, stdZ;  //!> standard deviation of values
        [[nodiscard]] EnuVector vector() const
        {
            Eigen::Matrix<double, kEnuSize, 1> vec;
            vec << x, y, z;
            return vec;
        }

        [[nodiscard]] EnuUncertainty uncertainty() const
        {
            Eigen::DiagonalMatrix<double, kEnuSize> uncertainty;
            // convert Standard Deviation  into Variance
            uncertainty.diagonal() << stdX * stdX, stdY * stdY, stdZ * stdZ;

            return uncertainty.toDenseMatrix();
        }
    };

    struct Orientation
    {
        Eigen::Quaterniond orientation;  //!> global orientation values (in ENU space)
        Eigen::Quaterniond uncertainty;  //!> standard deviation of values
    };

    //! Initialization with zero speed and acceleration, but at known ENU coordinate
    explicit PgaEKF(const Enu stationaryEnu) : _state(StateVector::Zero()), _uncertainty(UncertaintyMatrix::Identity())
    {
        _state[kMSC] = 1;
        _state[kM01] = -stationaryEnu.x / 2;
        _state[kM02] = -stationaryEnu.y / 2;
        _state[kM03] = -stationaryEnu.z / 2;

        Eigen::DiagonalMatrix<double, kStateSize> uncertainty;
        // variance is quadrupled because values are halved
        const double varX = stationaryEnu.stdX * stationaryEnu.stdX / 4;
        const double varY = stationaryEnu.stdY * stationaryEnu.stdY / 4;
        const double varZ = stationaryEnu.stdZ * stationaryEnu.stdZ / 4;
        uncertainty.diagonal() << 1., varX, varY, varZ, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        _uncertainty = uncertainty.toDenseMatrix();
    }

    //! Initialization with underlying state and uncertainty covariance matrix
    PgaEKF(StateVector initState, UncertaintyMatrix initUncertainty)
        : _state(std::move(initState)), _uncertainty(std::move(initUncertainty))
    {
    }

    //! Initialization with underlying state and uncertainty variance value
    PgaEKF(StateVector initState, double initUncertainty) : _state(std::move(initState))
    {
        _uncertainty = PgaEKF::UncertaintyMatrix::Identity() * initUncertainty;
    }

    //! @returns current State Vector
    [[nodiscard]] StateVector state() const { return _state; }

    //! @returns current Uncertainty Convariance matrix
    [[nodiscard]] UncertaintyMatrix uncertainty() const { return _uncertainty; }

    //! @returns current Position in ENU space
    [[nodiscard]] Enu filteredPosition() const
    {
        Enu enu{};
        enu.x = -_state[1] * 2;
        enu.y = -_state[2] * 2;
        enu.z = -_state[3] * 2;
        // std is also doubled
        enu.stdX = std::sqrt(_uncertainty.row(1)[1]) * 2;
        enu.stdY = std::sqrt(_uncertainty.row(2)[2]) * 2;
        enu.stdZ = std::sqrt(_uncertainty.row(3)[3]) * 2;
        return enu;
    }

    //! @returns current Orientation in ENU space
    [[nodiscard]] Orientation filteredOrientation() const
    {
        Orientation o{
            Eigen::Quaterniond(_state[0], -_state[6], -_state[5], -_state[4]),
            Eigen::Quaterniond(_uncertainty.row(0)[0], _uncertainty.row(6)[6], _uncertainty.row(5)[5], _uncertainty.row(4)[4])};
        return o;
    }

    //! Predicts current Kalman state and uncertainty into future
    //! @param dt - time delta for future state, must be > 0
    //! @param processNoise - noise/uncertainty of prediction
    void predict(const double dt, const double processNoiseStd = .01)
    {
        const auto noiseVariance = processNoiseStd * processNoiseStd;
        const ProcessNoiseMatrix Q = noiseVariance * ProcessNoiseMatrix::Identity();
        PredictJacobianMatrix F;
        _state = predictJacobian(dt, _state, F);
        _uncertainty = F * _uncertainty * F.transpose() + Q;
    }

    //! Updates state with given IMU-sensor data
    void updateImu(const Imu& imu)
    {
        ImuUncertainty R = imu.uncertainty();
        ImuUpdateJacobianMatrix H;
        auto h = ImuUpdateJacobian(_state, H);
        auto y = (imu.vector() - h).eval();                            // Innovation
        auto S = H * _uncertainty * H.transpose() + R;                 // Innovation covariance
        auto K = (_uncertainty * H.transpose() * S.inverse()).eval();  // Kalman Gate
        _state = _state + K * y;
        _uncertainty = (UncertaintyMatrix::Identity() - K * H) * _uncertainty;
    }

    //! Updates state with (external) position-sensor data
    void updateEnu(const Enu& enu)
    {
        EnuUncertainty R = enu.uncertainty();
        EnuUpdateJacobianMatrix H;
        auto h = EnuUpdateJacobian(_state, H);
        auto y = (enu.vector() - h).eval();                            // Innovation
        auto S = (H * _uncertainty * H.transpose() + R).eval();        // Innovation covariance
        auto K = (_uncertainty * H.transpose() * S.inverse()).eval();  // Kalman Gate
        _state = _state + K * y;
        _uncertainty = (UncertaintyMatrix::Identity() - K * H) * _uncertainty;
    }

    double motorNorm()
    {
        return std::sqrt(_state[6] * _state[6] + _state[5] * _state[5] + _state[4] * _state[4] + _state[0] * _state[0]);
    }

  private:
    using PredictJacobianMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;
    using ImuUpdateJacobianMatrix = Eigen::Matrix<double, kImuSize, kStateSize>;
    using EnuUpdateJacobianMatrix = Eigen::Matrix<double, kEnuSize, kStateSize>;

    StateVector _state;
    UncertaintyMatrix _uncertainty;

    static StateVector predictJacobian(const double dt, const StateVector& X, PredictJacobianMatrix& J)
    {
        StateVector S;
        // clang-format off
        S[kMSC] = (X[13] / 2.0 * X[6] + X[12] / 2.0 * X[5] + X[11] / 2.0 * X[4]) * dt + X[0];
        S[kM01] = ((X[13] / 2.0 * X[5] - X[0] / 2.0 * X[11]) * X[9] + (X[12] / 2.0 * X[5] + X[11] / 2.0 * X[4]) * X[8] - X[16] / 2.0 * X[5] + ((-(X[15] / 2.0)) - X[10] / 2.0 * X[13]) * X[4] - X[0] / 2.0 * X[14] - X[0] / 2.0 * X[10] * X[12]) * dt * dt + ((-(X[4] / 2.0 * X[9])) - X[0] / 2.0 * X[8] + X[13] / 2.0 * X[7] - X[10] / 2.0 * X[5] + X[12] / 2.0 * X[3] + X[11] / 2.0 * X[2]) * dt + X[1];
        S[kM02] = ((X[13] / 2.0 * X[6] + X[11] / 2.0 * X[4]) * X[9] + (X[12] / 2.0 * X[6] + X[0] / 2.0 * X[11]) * X[8] - X[16] / 2.0 * X[6] + (X[14] / 2.0 + X[10] / 2.0 * X[12]) * X[4] - X[0] / 2.0 * X[15] - X[0] / 2.0 * X[10] * X[13]) * dt * dt + ((-(X[0] / 2.0 * X[9])) + X[4] / 2.0 * X[8] - X[12] / 2.0 * X[7] - X[10] / 2.0 * X[6] + X[13] / 2.0 * X[3] - X[1] / 2.0 * X[11]) * dt + X[2];
        S[kM03] = ((X[11] / 2.0 * X[5] + X[0] / 2.0 * X[13]) * X[9] + (X[0] / 2.0 * X[12] - X[11] / 2.0 * X[6]) * X[8] + (X[15] / 2.0 + X[10] / 2.0 * X[13]) * X[6] + (X[14] / 2.0 + X[10] / 2.0 * X[12]) * X[5] - X[0] / 2.0 * X[16]) * dt * dt + (X[6] / 2.0 * X[9] + X[5] / 2.0 * X[8] + X[11] / 2.0 * X[7] - X[13] / 2.0 * X[2] - X[1] / 2.0 * X[12] - X[0] / 2.0 * X[10]) * dt + X[3];
        S[kM12] = ((-(X[12] / 2.0 * X[6])) + X[13] / 2.0 * X[5] - X[0] / 2.0 * X[11]) * dt + X[4];
        S[kM13] = (X[11] / 2.0 * X[6] - X[13] / 2.0 * X[4] - X[0] / 2.0 * X[12]) * dt + X[5];
        S[kM23] = ((-(X[11] / 2.0 * X[5])) + X[12] / 2.0 * X[4] - X[0] / 2.0 * X[13]) * dt + X[6];
        S[kM0123] = ((X[13] / 2.0 * X[4] - X[11] / 2.0 * X[6]) * X[9] + (X[12] / 2.0 * X[4] - X[11] / 2.0 * X[5]) * X[8] + ((-(X[14] / 2.0)) - X[10] / 2.0 * X[12]) * X[6] + (X[15] / 2.0 + X[10] / 2.0 * X[13]) * X[5] - X[16] / 2.0 * X[4]) * dt * dt + (X[5] / 2.0 * X[9] - X[6] / 2.0 * X[8] - X[10] / 2.0 * X[4] - X[11] / 2.0 * X[3] + X[12] / 2.0 * X[2] - X[1] / 2.0 * X[13]) * dt + X[7];
        S[kR01] = (X[11] * X[9] + X[14] + X[10] * X[12]) * dt + X[8];
        S[kR02] = ((-(X[11] * X[8])) + X[15] + X[10] * X[13]) * dt + X[9];
        S[kR03] = ((-(X[13] * X[9])) - X[12] * X[8] + X[16]) * dt + X[10];
        S[kR12] = X[kR12];
        S[kR13] = X[kR13];
        S[kR23] = X[kR23];
        S[kA01] = X[kA01];
        S[kA02] = X[kA02];
        S[kA03] = X[kA03];

        J.row(kMSC) << 1, 0, 0, 0, 0.5*X[11]*dt, 0.5*X[12]*dt, 0.5*X[13]*dt, 0, 0, 0, 0, 0.5*X[4]*dt, 0.5*X[5]*dt, 0.5*X[6]*dt, 0, 0, 0;
        J.row(kM01) << -0.5*X[8]*dt + dt*dt*(-0.5*X[10]*X[12] - 0.5*X[11]*X[9] - 0.5*X[14]), 1, 0.5*X[11]*dt, 0.5*X[12]*dt, -0.5*X[9]*dt + dt*dt*(-0.5*X[10]*X[13] + 0.5*X[11]*X[8] - 0.5*X[15]), -0.5*X[10]*dt + dt*dt*(0.5*X[12]*X[8] + 0.5*X[13]*X[9] - 0.5*X[16]), 0, 0.5*X[13]*dt, -0.5*X[0]*dt + dt*dt*(0.5*X[11]*X[4] + 0.5*X[12]*X[5]), -0.5*X[4]*dt + dt*dt*(-0.5*X[0]*X[11] + 0.5*X[13]*X[5]), -0.5*X[5]*dt + dt*dt*(-0.5*X[0]*X[12] - 0.5*X[13]*X[4]), 0.5* X[2] *dt + dt*dt*(-0.5*X[0]*X[9] + 0.5*X[4]*X[8]), 0.5*X[3]*dt + dt*dt*(-0.5*X[0]*X[10] + 0.5*X[5]*X[8]), 0.5*X[7]*dt + dt*dt*(-0.5*X[10]*X[4] + 0.5*X[5]*X[9]), -0.5*X[0]*dt*dt, -0.5*X[4]*dt*dt, -0.5*X[5]*dt*dt;
        J.row(kM02) << -0.5*X[9]*dt + dt*dt*(-0.5*X[10]*X[13] + 0.5*X[11]*X[8] - 0.5*X[15]), -0.5*X[11]*dt, 1, 0.5*X[13]*dt, 0.5*X[8]*dt + dt*dt*(0.5*X[10]*X[12] + 0.5*X[11]*X[9] + 0.5*X[14]), 0, -0.5*X[10]*dt + dt*dt*(0.5*X[12]*X[8] + 0.5*X[13]*X[9] - 0.5*X[16]), -0.5*X[12]*dt, 0.5*X[4]*dt + dt*dt*(0.5*X[0]*X[11] + 0.5*X[12]*X[6]), -0.5*X[0]*dt + dt*dt*(0.5*X[11]*X[4] + 0.5*X[13]*X[6]), -0.5*X[6]*dt + dt*dt*(-0.5*X[0]*X[13] + 0.5*X[12]*X[4]), -0.5*X[1]*dt + dt*dt*(0.5*X[0]*X[8] + 0.5*X[4]*X[9]), -0.5*X[7]*dt + dt*dt*(0.5*X[10]*X[4] + 0.5*X[6]*X[8]), 0.5*X[3]*dt + dt*dt*(-0.5*X[0]*X[10] + 0.5*X[6]*X[9]), 0.5*X[4]*dt*dt, -0.5*X[0]*dt*dt, -0.5*X[6]*dt*dt;
        J.row(kM03) << -0.5*X[10]*dt + dt*dt*(0.5*X[12]*X[8] + 0.5*X[13]*X[9] - 0.5*X[16]), -0.5*X[12]*dt, -0.5*X[13]*dt, 1, 0, 0.5*X[8]*dt + dt*dt*(0.5*X[10]*X[12] + 0.5*X[11]*X[9] + 0.5*X[14]), 0.5*X[9]*dt + dt*dt*(0.5*X[10]*X[13] - 0.5*X[11]*X[8] + 0.5*X[15]), 0.5*X[11]*dt, 0.5*X[5]*dt + dt*dt*(0.5*X[0]*X[12] - 0.5*X[11]*X[6]), 0.5*X[6]*dt + dt*dt*(0.5*X[0]*X[13] + 0.5*X[11]*X[5]), -0.5*X[0]*dt + dt*dt*(0.5*X[12]*X[5] + 0.5*X[13]*X[6]), 0.5*X[7]*dt + dt*dt*(0.5*X[5]*X[9] - 0.5*X[6]*X[8]), -0.5*X[1]*dt + dt*dt*(0.5*X[0]*X[8] + 0.5*X[10]*X[5]), -0.5*X[2]*dt + dt*dt*(0.5*X[0]*X[9] + 0.5*X[10]*X[6]), 0.5*X[5]*dt*dt, 0.5*X[6]*dt*dt, -0.5*X[0]*dt*dt;
        J.row(kM12) << -0.5*X[11]*dt, 0, 0, 0, 1, 0.5*X[13]*dt, -0.5*X[12]*dt, 0, 0, 0, 0, -0.5*X[0]*dt, -0.5*X[6]*dt, 0.5*X[5]*dt, 0, 0, 0;
        J.row(kM13) << -0.5*X[12]*dt, 0, 0, 0, -0.5*X[13]*dt, 1, 0.5*X[11]*dt, 0, 0, 0, 0, 0.5*X[6]*dt, -0.5*X[0]*dt, -0.5*X[4]*dt, 0, 0, 0;
        J.row(kM23) << -0.5*X[13]*dt, 0, 0, 0, 0.5*X[12]*dt, -0.5*X[11]*dt, 1, 0, 0, 0, 0, -0.5*X[5]*dt, 0.5*X[4]*dt, -0.5*X[0]*dt, 0, 0, 0;
        J.row(kM0123) << 0, -0.5*X[13]*dt, 0.5*X[12]*dt, -0.5*X[11]*dt, -0.5*X[10]*dt + dt*dt*(0.5*X[12]*X[8] + 0.5*X[13]*X[9] - 0.5*X[16]), 0.5*X[9]*dt + dt*dt*(0.5*X[10]*X[13] - 0.5*X[11]*X[8] + 0.5*X[15]), -0.5*X[8]*dt + dt*dt*(-0.5*X[10]*X[12] - 0.5*X[11]*X[9] - 0.5*X[14]), 1, -0.5*X[6]*dt + dt*dt*(-0.5*X[11]*X[5] + 0.5*X[12]*X[4]), 0.5*X[5]*dt + dt*dt*(-0.5*X[11]*X[6] + 0.5*X[13]*X[4]), -0.5*X[4]*dt + dt*dt*(-0.5*X[12]*X[6] + 0.5*X[13]*X[5]), -0.5*X[3]*dt + dt*dt*(-0.5*X[5]*X[8] - 0.5*X[6]*X[9]), 0.5*X[2]*dt + dt*dt*(-0.5*X[10]*X[6] + 0.5*X[4]*X[8]), -0.5*X[1]*dt + dt*dt*(0.5*X[10]*X[5] + 0.5*X[4]*X[9]), -0.5*X[6]*dt*dt, 0.5*X[5]*dt*dt, -0.5*X[4]*dt*dt;
        J.row(kR01) << 0, 0, 0, 0, 0, 0, 0, 0, 1, X[11]*dt, X[12]*dt, X[9]*dt, X[10]*dt, 0, dt, 0, 0;
        J.row(kR02) << 0, 0, 0, 0, 0, 0, 0, 0, -X[11]*dt, 1, X[13]*dt, -X[8]*dt, 0, X[10]*dt, 0, dt, 0;
        J.row(kR03) << 0, 0, 0, 0, 0, 0, 0, 0, -X[12]*dt, -X[13]*dt, 1, 0, -X[8]*dt, -X[9]*dt, 0, 0, dt;
        J.row(kR12) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
        J.row(kR13) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
        J.row(kR23) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
        J.row(kA01) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0;
        J.row(kA02) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
        J.row(kA03) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
        // clang-format on
        return S;
    }

    static ImuVector ImuUpdateJacobian(const StateVector& X, ImuUpdateJacobianMatrix& J)
    {
        ImuVector H;
        H[0] = -(2.0 * X[4] * X[6] - 2.0 * X[0] * X[5]) * kGravity + X[14];
        H[1] = -((-(2.0 * X[0] * X[6])) - 2.0 * X[4] * X[5]) * kGravity + X[15];
        H[2] = -((-(X[6] * X[6])) - X[5] * X[5] + X[4] * X[4] + X[0] * X[0]) * kGravity + X[16];
        H[3] = X[11];
        H[4] = X[12];
        H[5] = X[13];

        J.row(0) << 2.0 * X[5] * kGravity, 0, 0, 0, -2.0 * X[6] * kGravity, 2.0 * X[0] * kGravity, -2.0 * X[4] * kGravity, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0;
        J.row(1) << 2.0 * X[6] * kGravity, 0, 0, 0, 2.0 * X[5] * kGravity, 2.0 * X[4] * kGravity, 2.0 * X[0] * kGravity, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0;
        J.row(2) << -2 * X[0] * kGravity, 0, 0, 0, -2 * X[4] * kGravity, 2 * X[5] * kGravity, 2 * X[6] * kGravity, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1;
        J.row(3) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
        J.row(4) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
        J.row(5) << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;

        return H;
    }

    static EnuVector EnuUpdateJacobian(const StateVector& X, EnuUpdateJacobianMatrix& J)
    {
        EnuVector H;
        H[0] = -2 * X[1];
        H[1] = -2 * X[2];
        H[2] = -2 * X[3];

        J.row(0) << 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        J.row(1) << 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        J.row(2) << 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;

        return H;
    }
};

}  // namespace pga_ekf

#endif  // PGA_EKF_PGA_EKF_H