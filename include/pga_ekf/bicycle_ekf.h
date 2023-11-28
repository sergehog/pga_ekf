// EKF for Bicycle Vehicle Model (aka Ackermann steering model)
// Created by sergey on 28.11.2023.
//

#ifndef PGA_EKF_BICYCLE_EKF_H
#define PGA_EKF_BICYCLE_EKF_H
#include <Eigen/Dense>
#include <iostream>
#include <utility>
#include "base_ekf.h"

namespace pga_ekf
{

namespace bicycle
{

//! Named indexes of the State Vector
//! Intentionally made as a enum, such that visibility is inside of the namespace
enum StateIndex : std::size_t
{
    /// 4 components of the 2D PGA Motor,
    /// Encodes Body Frame Pose in some global (ENU-like) coordinate frame
    kMSC = 0,
    kM01 = 1,
    kM02 = 2,
    kM12 = 3,
    kSteeringAngle = 4,
    kVelocity = 5,
    kSteeringVelocity = 6,
    kAcceleration = 7,
    kStateSize = 8
};

//! Gravity constant
constexpr double kGravity = 9.80665;

//! EKF filter for 2 types of inputs:
//! - IMU data in body-frame
//! - ENU Position in world-frame - can be obtained from GPS or some other sensor
//! We loosely call global coordinate frame "ENU",
//! however, only real constraint for it is that last "U" coordinate points up

template<typename ScalarType = double>
class BicycleEKF : public BaseEkf<BicycleEKF<ScalarType>, ScalarType, StateIndex::kStateSize>
{
  public:
    static constexpr std::size_t kStateSize = StateIndex::kStateSize;
    using StateVector = typename BaseEkf<BicycleEKF<ScalarType>, ScalarType, StateIndex::kStateSize>::StateVector;
    using UncertaintyMatrix = typename BaseEkf<BicycleEKF<ScalarType>, ScalarType, StateIndex::kStateSize>::UncertaintyMatrix;
    using ProcessNoiseMatrix = UncertaintyMatrix;
    using PredictJacobianMatrix = UncertaintyMatrix;

    const ScalarType wheelBase;
    BicycleEKF(const ScalarType _wheelBase = 1.) : wheelBase(_wheelBase)
    {}

//    static constexpr std::size_t kImuSize = 6;
//    static constexpr std::size_t kEnuSize = 3;
//    using StateVector = Eigen::Matrix<double, kStateSize, 1>;
//    using ImuVector = Eigen::Matrix<double, kImuSize, 1>;
//    using EnuVector = Eigen::Matrix<double, kEnuSize, 1>;
//
//    using UncertaintyMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;
//    using ProcessNoiseMatrix = Eigen::Matrix<double, kStateSize, kStateSize>;
//    using ImuUncertainty = Eigen::Matrix<double, kImuSize, kImuSize>;
//    using EnuUncertainty = Eigen::Matrix<double, kEnuSize, kEnuSize>;

  protected:

    StateVector predictJacobian(const ScalarType dt, const ProcessNoiseMatrix& Q, PredictJacobianMatrix& P)
    {
        constexpr auto X = this->_state;
        StateVector S;
        //Rate = (e12*sin(angle) - e01*L*cos(angle)) * velocity;
        //M(t+dt) = M(t) + 0.5 * dt * M(t) * Rate;
	S[kMSC] = X[kMSC] - X[kM12] / 2.0 * sin(X[kSteeringAngle]) * dt * X[kVelocity];
	S[kM01] = ((-(X[kM02] / 2.0 * sin(X[kSteeringAngle]))) - wheelBase / 2.0 * X[kMSC] * cos(X[kSteeringAngle])) * dt * X[kVelocity] + X[kM01];
	S[kM02] = (X[kM01] / 2.0 * sin(X[kSteeringAngle]) + wheelBase / 2.0 * X[kM12] * cos(X[kSteeringAngle])) * dt * X[kVelocity] + X[kM02];
	S[kM12] = X[kMSC] / 2.0 * sin(X[kSteeringAngle]) * dt * X[kVelocity] + X[kM12];
        S[kSteeringAngle] = X[kSteeringAngle] + X[kSteeringVelocity] * dt;
        S[kVelocity] = X[kVelocity] + X[kAcceleration] * dt;
        S[kSteeringVelocity] = X[kSteeringVelocity];
        S[kAcceleration] = X[kAcceleration];


    }

};

}  // namespace bicycle

} // namespace pga_ekf

#endif  // PGA_EKF_BICYCLE_EKF_H
