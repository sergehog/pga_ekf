//
// Created by sergey on 22.10.2022.
//

#ifndef PGA_EKF_EKF_H
#define PGA_EKF_EKF_H

#include <Eigen/Dense>
#include <cstddef>

namespace pga_ekf
{

template <class DerivedEfk, typename ScalarType, std::size_t kStateSize>
class BaseEkf
{
  public:
    using StateVector = Eigen::Matrix<ScalarType, kStateSize, 1>;
    using UncertaintyMatrix = Eigen::Matrix<ScalarType, kStateSize, kStateSize>;
    using ProcessNoiseMatrix = UncertaintyMatrix;
    using PredictJacobianMatrix = UncertaintyMatrix;

    //! @returns current State Vector
    [[nodiscard]] StateVector state() const { return _state; }

    //! @returns current Uncertainty Convariance matrix
    [[nodiscard]] UncertaintyMatrix uncertainty() const { return _uncertainty; }

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

  protected:
    StateVector _state;
    UncertaintyMatrix _uncertainty;

    //! Predicts current Kalman state and uncertainty into future
    void predict(const ScalarType dt, const ProcessNoiseMatrix& Q)
    {
        PredictJacobianMatrix F;
        _state = static_cast<DerivedEfk*>(this)->predictJacobian(dt, _state, F);
        _uncertainty = F * _uncertainty * F.transpose() + Q;
    }


    //! Generic Update function, use template parameter ID to distinguish between different update variants
//    template <std::size_t kObservationsAmnt, std::size_t ID=0>
//    void update(const Eigen::Matrix<ScalarType, kObservationsAmnt, 1>& observations,
//                const Eigen::Matrix<ScalarType, kObservationsAmnt, kObservationsAmnt>& uncertainty)
//    {
//        using JacobianMatrix = Eigen::Matrix<ScalarType, kObservationsAmnt, kObservationsAmnt>;
//        JacobianMatrix H;
//        Eigen::Matrix<ScalarType, kObservationsAmnt, 1> h =
//            static_cast<DerivedEfk*>(this)->updateJacobian<kObservationsAmnt, ID>(_state, H);  // predictions
//        Eigen::Matrix<ScalarType, kObservationsAmnt, 1> y = (observations - h).eval();         // Innovation
//        auto S = H * _uncertainty * H.transpose() + uncertainty;                               // Innovation covariance
//        auto K = (_uncertainty * H.transpose() * S.inverse()).eval();                          // Kalman Gate
//        _state = _state + K * y;
//        _uncertainty = (UncertaintyMatrix::Identity() - K * H) * _uncertainty;
//    }
};

}  // namespace pga_ekf

#endif  // PGA_EKF_EKF_H
