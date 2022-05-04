#include <gtest/gtest.h>
#include <pga_ekf/pga_ekf.h>
#include <array>
#include <iostream>
using namespace pga_ekf;

TEST(PgaEKFTest, BasicTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    // Check default values
    EXPECT_EQ(ekf.filteredPosition().x, 1);
    EXPECT_EQ(ekf.filteredPosition().y, 2);
    EXPECT_EQ(ekf.filteredPosition().z, 3);
    EXPECT_EQ(ekf.filteredPosition().stdX, 4);
    EXPECT_EQ(ekf.filteredPosition().stdY, 5);
    EXPECT_EQ(ekf.filteredPosition().stdZ, 6);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    // orientation is unknown -> uncertainty == 1
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.x(), 1);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.y(), 1);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1);
}

//! when speed=0, acc=0, position does not change, but uncertanty grows
TEST(PgaEKFTest, BasicPredictTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    const double processNoise = .02;
    ekf.predict(0.1, processNoise);
    // Check default values
    EXPECT_EQ(ekf.filteredPosition().x, 1);
    EXPECT_EQ(ekf.filteredPosition().y, 2);
    EXPECT_EQ(ekf.filteredPosition().z, 3);
    EXPECT_EQ(ekf.filteredPosition().stdX, 4 + processNoise * 2);
    EXPECT_EQ(ekf.filteredPosition().stdY, 5 + processNoise * 2);
    EXPECT_EQ(ekf.filteredPosition().stdZ, 6 + processNoise * 2);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    // orientaiton is unknown -> uncertainty == 1
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.w(), 1 + processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.x(), 1 + processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.y(), 1 + processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1 + processNoise);
}

//! Tests PgaEKF.predict(..) function
//! @param tuple[0] - Initial (input) state
//! @param tuple[1] - Expected (output) state
class PgaEKF_PredictTest
    : public ::testing::TestWithParam<std::tuple<std::array<double, PgaEKF::kStateSize>, std::array<double, PgaEKF::kStateSize>>>
{
  public:
    constexpr static double kProcessNoise = 0.1;
    constexpr static double kVelocity = 1;
    constexpr static double kAccel = 1;
    constexpr static double kTimeDelta = 0.1;
    constexpr static double kAccuracy = 1e-10;

    constexpr static std::array<double, PgaEKF::kStateSize> kStationaryIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kStationaryExpected = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kStationary2In = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0,
                                                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kStationary2Expected = {1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0,
                                                                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kXVelocityIn = {1, 0, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0,
                                                                            0, 0, 0, 0, 0, 0, 0, 0, 0,         0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kXVelocityExpected = {
        1, -kVelocity* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kYVelocityIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0,
                                                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,         0};
    constexpr static std::array<double, PgaEKF::kStateSize> kYVelocityExpected = {
        1, 0, -kVelocity* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kZVelocityIn = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, kVelocity, 0,
                                                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kZVelocityExpected = {
        1, 0, 0, -kVelocity* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kVelocity, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kXAccelerationIn = {1, 0, 0,      0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                0, 0, kAccel, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kXAccelerationExpected = {
        1, -kAccel* kTimeDelta* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kAccel* kTimeDelta, 0, 0, 0, 0, 0, kAccel, 0, 0, 0, 0, 0, 0, 0,
        0};
    constexpr static std::array<double, PgaEKF::kStateSize> kYAccelerationIn = {1, 0, 0, 0,      0, 0, 0, 0, 0, 0, 0, 0,
                                                                                0, 0, 0, kAccel, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kYAccelerationExpected = {
        1, 0, -kAccel* kTimeDelta* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kAccel* kTimeDelta, 0, 0, 0, 0, 0, kAccel, 0, 0, 0, 0, 0,
        0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kZAccelerationIn = {1, 0, 0, 0, 0,      0, 0, 0, 0, 0, 0, 0,
                                                                                0, 0, 0, 0, kAccel, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> kZAccelerationExpected = {
        1, 0, 0, -kAccel* kTimeDelta* kTimeDelta / 2, 0, 0, 0, 0, 0, 0, kAccel* kTimeDelta, 0, 0, 0, 0, 0, kAccel, 0, 0, 0,
        0, 0, 0};

  public:
    std::array<double, PgaEKF::kStateSize> _expectedState{};
    PgaEKF::StateVector _inputState;

    PgaEKF_PredictTest()
    {
        std::array<double, PgaEKF::kStateSize> input{};
        std::tie(input, _expectedState) = GetParam();

        // convert array<...> into actual State vector
        for (std::size_t i = 0UL; i < PgaEKF::kStateSize; i++)
        {
            _inputState[i] = input[i];
        }
    }
};

TEST_P(PgaEKF_PredictTest, ParametrizedPredictTest)
{
    PgaEKF::UncertaintyMatrix uncertainty = PgaEKF::UncertaintyMatrix::Identity() * kAccuracy;

    PgaEKF ekf(_inputState, uncertainty);
    ekf.predict(kTimeDelta, kProcessNoise);

    for (std::size_t i = 0UL; i < PgaEKF::kStateSize; i++)
    {
        EXPECT_NEAR(ekf.state()[i], _expectedState[i], kAccuracy) << ", i=" << i;
    }

    for (std::size_t i = 0UL; i < PgaEKF::kStateSize; i++)
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

//! Checks that PgaEKF::updateImu works at all
TEST(PgaEKFTest, BasicUpdateImuTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    PgaEKF::Imu imu{0, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    ekf.updateImu(imu);

    // After one IMU update orientation uncertainty decreased
    // everything else stay the same
    EXPECT_EQ(ekf.filteredPosition().x, 1);
    EXPECT_EQ(ekf.filteredPosition().y, 2);
    EXPECT_EQ(ekf.filteredPosition().z, 3);
    EXPECT_EQ(ekf.filteredPosition().stdX, 4);
    EXPECT_EQ(ekf.filteredPosition().stdY, 5);
    EXPECT_EQ(ekf.filteredPosition().stdZ, 6);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    EXPECT_LT(ekf.filteredOrientation().uncertainty.w(), .5);
    EXPECT_LT(ekf.filteredOrientation().uncertainty.x(), .003);
    EXPECT_LT(ekf.filteredOrientation().uncertainty.y(), .003);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1);
}

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
    constexpr static std::array<double, PgaEKF::kStateSize> kStationaryExpected = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static PgaEKF::Imu kStationaryImu = {0, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};

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

TEST_P(PgaEKF_UpdateImuTest, ParametrizedPredictTest)
{
    PgaEKF::UncertaintyMatrix uncertainty = PgaEKF::UncertaintyMatrix::Identity() * kAccuracy;

    PgaEKF ekf(_inputState, uncertainty);

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
                                                        PgaEKF_UpdateImuTest::kStationaryExpected)));

// TEST(PgaEKFTest, AccPositiveXTest)
//{
//     PgaEKF::Enu enu{0, 0, 0, 0.001, 0.001, 0.001};
//     PgaEKF ekf(enu);
//     std::cout << "Initial State: " << ekf.state().transpose() << std::endl;
//     std::cout << "Initial Uncertainty: " << std::endl;
//     std::cout << ekf.uncertainty() << std::endl << std::endl;
//     //    EXPECT_NEAR(ekf.uncertainty().row(5)[5], 1, 1e-10);
//     //    EXPECT_NEAR(ekf.uncertainty().row(6)[6], 1, 1e-10);
//     //    EXPECT_NEAR(ekf.uncertainty().row(7)[7], 1, 1e-10);
//     PgaEKF::Imu imu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
//     ekf.updateImu(imu);
//     //    EXPECT_NEAR(ekf.uncertainty().row(5)[5], 1, 1e-10);
//     //    EXPECT_LT(ekf.uncertainty().row(6)[6], 0.003);
//     //    EXPECT_LT(ekf.uncertainty().row(7)[7], 0.003);
//
//     std::cout << "State after IMU update: " << ekf.state().transpose() << std::endl;
//     std::cout << "Uncertainty after IMU update: " << std::endl;
//     std::cout << ekf.uncertainty() << std::endl << std::endl;
// }

// TEST(PgaEKFTest, MovingXPositiveTest)
//{
//     PgaEKF::Enu enu{0, 0, 0, 0.001, 0.001, 0.001};
//     PgaEKF ekf(enu);
//
//     std::cout << "Initial State: " << ekf.state().transpose() << std::endl;
//     PgaEKF::Imu _inputImu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
//     ekf.updateImu(_inputImu);
//     std::cout << "State after Imu Update: " << ekf.state().transpose() << std::endl;
// }

// TEST(PgaEKFTest, UncertaintyGrowDuringAccelerationTest)
//{
//     PgaEKF::Enu enu{0, 0, 0, 1e-8, 1e-8, 1e-8};
//     PgaEKF ekf(enu);
//     std::cout << "State before Imu Update: " << ekf.state().transpose() << std::endl;
//     PgaEKF::Imu _inputImu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10};
//     ekf.updateImu(_inputImu);
//     std::cout << "State after Imu Update: " << ekf.state().transpose() << std::endl;
//
//     ekf.predict(0.1, 1e-8);
//
//     std::cout << "State after Predict: " << ekf.state().transpose() << std::endl;
//
//     // Orientation Uncertainty Grows During Acceleration
//     EXPECT_GE(ekf.uncertainty().row(8)[8], 0.01);
//     EXPECT_GE(ekf.uncertainty().row(9)[9], 0.01);
//     EXPECT_GE(ekf.uncertainty().row(10)[10], 0.01);
////    std::cout << "Uncertainty after predict: " << std::endl;
////    std::cout << ekf.uncertainty() << std::endl << std::endl;
//
//    // Angular Momentum uncertainty stays zero
////    EXPECT_NEAR(ekf.uncertainty().row(11)[11], 0., 1e-7);
////    EXPECT_NEAR(ekf.uncertainty().row(12)[12], 0., 1e-7);
////    EXPECT_NEAR(ekf.uncertainty().row(13)[13], 0., 1e-7);
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