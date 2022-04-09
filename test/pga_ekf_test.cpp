#include <iostream>
#include <gtest/gtest.h>
#include <pga_ekf/pga_ekf.h>
#include <array>
using namespace pga_ekf;

TEST(PgaEKFTest, BasicTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    // Check default values
    EXPECT_EQ(ekf.filteredEnu().x, 1);
    EXPECT_EQ(ekf.filteredEnu().y, 2);
    EXPECT_EQ(ekf.filteredEnu().z, 3);
    EXPECT_EQ(ekf.filteredEnu().stdX, 4);
    EXPECT_EQ(ekf.filteredEnu().stdY, 5);
    EXPECT_EQ(ekf.filteredEnu().stdZ, 6);

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
    EXPECT_EQ(ekf.filteredEnu().x, 1);
    EXPECT_EQ(ekf.filteredEnu().y, 2);
    EXPECT_EQ(ekf.filteredEnu().z, 3);
    EXPECT_EQ(ekf.filteredEnu().stdX, 4+processNoise*2);
    EXPECT_EQ(ekf.filteredEnu().stdY, 5+processNoise*2);
    EXPECT_EQ(ekf.filteredEnu().stdZ, 6+processNoise*2);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    // orientaiton is unknown -> uncertainty == 1
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.w(), 1+processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.x(), 1+processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.y(), 1+processNoise);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1+processNoise);
}

class EkfPredictTest : public ::testing::TestWithParam<std::tuple<std::array<double, PgaEKF::kStateSize>, std::array<double, PgaEKF::kStateSize>>>
{
  public:

    constexpr static double processNoise = 0.1;
    constexpr static double velocity = 1;
    constexpr static double acceleration = 1;
    constexpr static double timeDelta = 0.1;

    constexpr static std::array<double, PgaEKF::kStateSize> StationaryIn = {1, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> StationaryExpected = {1, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> Stationary2In = {1, 2, 3, 4, 5, 6, 7, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> Stationary2Expected = {1, 2, 3, 4, 5, 6, 7, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> XVelocityIn =       {1, 0, 0, 0, 0, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> XVelocityExpected = {1, -velocity*timeDelta/2, 0, 0, 0, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> YVelocityIn =       {1, 0, 0, 0, 0, 0, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> YVelocityExpected = {1, 0, -velocity*timeDelta/2, 0, 0, 0, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> ZVelocityIn =       {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> ZVelocityExpected = {1, 0, 0, -velocity*timeDelta/2, 0, 0, 0, 0, 0, 0, velocity, 0, 0, 0, 0, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> XAccelerationIn =       {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, acceleration, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> XAccelerationExpected = {1, -acceleration*timeDelta*timeDelta/2, 0, 0, 0, 0, 0, 0, acceleration*timeDelta, 0, 0, 0, 0, 0, acceleration, 0, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> YAccelerationIn =       {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, acceleration, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> YAccelerationExpected = {1, 0, -acceleration*timeDelta*timeDelta/2, 0, 0, 0, 0, 0, 0, acceleration*timeDelta, 0, 0, 0, 0, 0, acceleration, 0};
    constexpr static std::array<double, PgaEKF::kStateSize> ZAccelerationIn =       {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, acceleration};
    constexpr static std::array<double, PgaEKF::kStateSize> ZAccelerationExpected = {1, 0, 0, -acceleration*timeDelta*timeDelta/2, 0, 0, 0, 0, 0, 0, acceleration*timeDelta, 0, 0, 0, 0, 0, acceleration};

    EkfPredictTest() = default;
};

TEST_P(EkfPredictTest, ParametrizedPredictTest)
{
    std::array<double, PgaEKF::kStateSize> input{};
    std::array<double, PgaEKF::kStateSize> expected{};
    PgaEKF::StateVector inputState;

    std::tie(input, expected) = GetParam();
    for(std::size_t i=0; i<PgaEKF::kStateSize; i++)
    {
        inputState[i] = input[i];
    }
    const double accuracy = 1e-10;
    PgaEKF::UncertaintyMatrix uncertainty = PgaEKF::UncertaintyMatrix::Identity() * 1e-10;
    PgaEKF ekf(inputState, uncertainty);
    ekf.predict(timeDelta, processNoise);

    for(std::size_t i=0; i<PgaEKF::kStateSize; i++)
    {
        EXPECT_NEAR(ekf.state()[i], expected[i], accuracy) << ", i=" << i;
    }

    for(std::size_t i = 0; i<PgaEKF::kStateSize; i++)
    {
        EXPECT_NEAR(ekf.uncertainty().row(i)[i], processNoise+1e-10, 1e-10);
    }
}

INSTANTIATE_TEST_CASE_P(InstantiationName,
                        EkfPredictTest,
                        testing::Values(std::make_tuple(EkfPredictTest::StationaryIn, EkfPredictTest::StationaryExpected),
                                        std::make_tuple(EkfPredictTest::Stationary2In, EkfPredictTest::Stationary2Expected),
                                        std::make_tuple(EkfPredictTest::XVelocityIn, EkfPredictTest::XVelocityExpected),
                                        std::make_tuple(EkfPredictTest::YVelocityIn, EkfPredictTest::YVelocityExpected),
                                        std::make_tuple(EkfPredictTest::ZVelocityIn, EkfPredictTest::ZVelocityExpected),
                                        std::make_tuple(EkfPredictTest::XAccelerationIn, EkfPredictTest::XAccelerationExpected),
                                        std::make_tuple(EkfPredictTest::YAccelerationIn, EkfPredictTest::YAccelerationExpected),
                                        std::make_tuple(EkfPredictTest::ZAccelerationIn, EkfPredictTest::ZAccelerationExpected)
                                        ));



TEST(PgaEKFTest, ImuUpdateTest)
{
    PgaEKF::Enu enu{1, 2, 3, 4, 5, 6};
    PgaEKF ekf(enu);
    PgaEKF::Imu imu{0, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    ekf.updateImu(imu);

    // After one IMU update orientation uncertainty decreased
    // everything else stay the same
    EXPECT_EQ(ekf.filteredEnu().x, 1);
    EXPECT_EQ(ekf.filteredEnu().y, 2);
    EXPECT_EQ(ekf.filteredEnu().z, 3);
    EXPECT_EQ(ekf.filteredEnu().stdX, 4);
    EXPECT_EQ(ekf.filteredEnu().stdY, 5);
    EXPECT_EQ(ekf.filteredEnu().stdZ, 6);

    EXPECT_EQ(ekf.filteredOrientation().orientation.w(), 1);
    EXPECT_EQ(ekf.filteredOrientation().orientation.x(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.y(), 0);
    EXPECT_EQ(ekf.filteredOrientation().orientation.z(), 0);

    EXPECT_LT(ekf.filteredOrientation().uncertainty.w(), .5);
    EXPECT_LT(ekf.filteredOrientation().uncertainty.x(), .5);
    EXPECT_LT(ekf.filteredOrientation().uncertainty.y(), .5);
    EXPECT_EQ(ekf.filteredOrientation().uncertainty.z(), 1);
}

TEST(PgaEKFTest, AccPositiveXTest)
{
    PgaEKF::Enu enu{0, 0, 0, 0.001, 0.001, 0.001};
    PgaEKF ekf(enu);

    std::cout << "Initial State: " << ekf.state().transpose() << std::endl;
    PgaEKF::Imu imu{1, 0, pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    ekf.updateImu(imu);
    std::cout << "State after Imu Update: " << ekf.state().transpose() << std::endl;
}

//TEST(PgaEKFTest, UncertaintyGrowDuringAccelerationTest)
//{
//    PgaEKF::Enu enu{0, 0, 0, 1e-8, 1e-8, 1e-8};
//    PgaEKF ekf(enu);
//    std::cout << "State before Imu Update: " << ekf.state().transpose() << std::endl;
//    PgaEKF::Imu imu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 1e-8, 1e-8, 1e-8, 1e-10, 1e-10, 1e-10};
//    ekf.updateImu(imu);
//    std::cout << "State after Imu Update: " << ekf.state().transpose() << std::endl;
//
//    ekf.predict(0.1, 1e-8);
//
//    std::cout << "State after Predict: " << ekf.state().transpose() << std::endl;
//
//    // Orientation Uncertainty Grows During Acceleration
//    EXPECT_GE(ekf.uncertainty().row(8)[8], 0.01);
//    EXPECT_GE(ekf.uncertainty().row(9)[9], 0.01);
//    EXPECT_GE(ekf.uncertainty().row(10)[10], 0.01);
////    std::cout << "Uncertainty after predict: " << std::endl;
////    std::cout << ekf.uncertainty() << std::endl << std::endl;
//
//    // Angular Momentum uncertainty stays zero
////    EXPECT_NEAR(ekf.uncertainty().row(11)[11], 0., 1e-7);
////    EXPECT_NEAR(ekf.uncertainty().row(12)[12], 0., 1e-7);
////    EXPECT_NEAR(ekf.uncertainty().row(13)[13], 0., 1e-7);
//}



//TEST(PgaEKFTest, MoveTowardsPositiveXTest)
//{
//    PgaEKF::Enu enu{0, 0, 0, 0.001, 0.001, 0.001};
//    PgaEKF ekf(enu);
//
//    std::cout << "Initial State: " << ekf.state().transpose() << std::endl;
//
//    PgaEKF::Imu imu{1, 0, -pga_ekf::kGravity, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
//
//    for(int i=0; i<2; i++)
//    {
//        std::cout << "==================== IMU ===========================" << std::endl;
//        ekf.updateImu(imu);
//        std::cout << "State after update: " << ekf.state().transpose() << std::endl;
//        std::cout << "Uncertainty after update: " << std::endl;
//        std::cout << ekf.uncertainty() << std::endl << std::endl;
//        ekf.predict(0.1);
//        std::cout << "State after predict: " << ekf.state().transpose() << std::endl;
//        std::cout << "Uncertainty after predict: " << std::endl;
//        std::cout << ekf.uncertainty() << std::endl << std::endl;
//    }
//
//    std::cout << "================== ENU =============================" << std::endl;
//    enu.x = 0.01;
//    //ekf.updateEnu(enu);
//
//}

//        std::cout << "Position: " << ekf.filteredEnu().x << "," << ekf.filteredEnu().y << "," << ekf.filteredEnu().z
//                  << std::endl;
//        std::cout << "Position uncertainty: " << ekf.filteredEnu().stdX << "," << ekf.filteredEnu().stdY << ","
//                  << ekf.filteredEnu().stdZ << std::endl;
//        std::cout << "Orientation: " << ekf.filteredOrientation().orientation.coeffs().transpose() << std::endl;
//        std::cout << "Orientation uncertainty: " << ekf.filteredOrientation().uncertainty.coeffs().transpose()
//                  << std::endl;