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

#include "OpenZen.h"
#include "pga_ekf/pga_ekf.h"
#include <iostream>

using namespace zen;
using namespace pga_ekf;

PgaEKF::Imu estimateMeanAndStdDev(ZenClient& client, size_t samples = 200);

int main(int argc, char* argv[])
{
    // create OpenZen Client
    auto [clientError, client] = make_client();

    if (clientError)
    {
        std::cout << "Cannot create OpenZen client" << std::endl;
        return clientError;
    }

    auto [obtainError, sensor] = client.obtainSensorByName("Bluetooth", "00:04:3E:4B:31:94");

    if (obtainError)
    {
        std::cout << "Cannot connect to sensor" << std::endl;
        client.close();
        return obtainError;
    }

    // check that the sensor has an IMU component
    auto imuSensor = sensor.getAnyComponentOfType(g_zenSensorType_Imu);

    if (!imuSensor)
    {
        std::cout << "Connected sensor has no IMU" << std::endl;
        client.close();
        return ZenError_WrongSensorType;
    }

    // set and get current streaming frequency
    auto error = imuSensor->setInt32Property(ZenImuProperty_SamplingRate, 200);
    if (error)
    {
        std::cout << "Error setting streaming frequency" << std::endl;
        client.close();
        return error;
    }

    auto freqPair = imuSensor->getInt32Property(ZenImuProperty_SamplingRate);
    if (freqPair.first)
    {
        std::cout << "Error fetching streaming frequency" << std::endl;
        client.close();
        return freqPair.first;
    }
    std::cout << "Streaming frequency: " << freqPair.second << std::endl;
    const double dt = 1.0 / double(freqPair.second);

    // toggle on/off of a particular data output (linAcc is not ON by default)
    error = imuSensor->setBoolProperty(ZenImuProperty_OutputLinearAcc, true);
    if (error)
    {
        std::cout << "Error toggling ON linear acc data output" << std::endl;
        client.close();
        return error;
    }

    PgaEKF::Imu mean = estimateMeanAndStdDev(client, 200);

    std::cout << "stdAx=" << mean.stdAx << "; stdAy=" << mean.stdAy << "; stdAz=" << mean.stdAz;
    std::cout << "; stdGx=" << mean.stdGx << "; stdGy=" << mean.stdGy << "; stdGz=" << mean.stdGz << std::endl;

    PgaEKF filter(PgaEKF::Enu{0, 0, 0});

    for (size_t i = 0; i < 200; i++)
    {
        auto event = client.waitForNextEvent();
        if (event.has_value() && event->eventType == ZenEventType_ImuData)
        {
            auto imuInput = event->data.imuData;
            PgaEKF::Imu value = mean;
            value.ax = imuInput.a[0] * kGravity;
            value.ay = imuInput.a[1] * kGravity;
            value.az = imuInput.a[2] * kGravity;
            value.gx = imuInput.g1[0];
            value.gy = imuInput.g1[1];
            value.gz = imuInput.g1[2];

            filter.predict(dt);
            filter.updateImu(value);
            std::cout << "ax=" << value.ax << "; ay=" << value.ay << "; az=" << value.az;
            std::cout << "; gx=" << value.gx << "; gy=" << value.gy << "; gz=" << value.gz;
            auto enu = filter.filteredPosition();
            auto q = filter.filteredOrientation().orientation.coeffs();
            std::cout << "; [X=" << enu.x << "; Y=" << enu.y << "; Z=" << enu.z << "]";
            std::cout << ";  Q:(x=" << q.x() << ", y=" << q.y() << ", z=" << q.z() << ", w=" << q.w() << ")" << std::endl;
        }
    }

    client.close();

    return 0;
}

PgaEKF::Imu estimateMeanAndStdDev(ZenClient& client, size_t samples)
{
    // Estimate std-devs for IMU using some measurements
    std::vector<PgaEKF::Imu> imuValues{};
    PgaEKF::Imu meanValue;
    for (size_t i = 0; i < samples; i++)
    {
        auto event = client.waitForNextEvent();
        if (event.has_value() && event->eventType == ZenEventType_ImuData)
        {
            auto imuInput = event->data.imuData;

            PgaEKF::Imu imuValue{imuInput.a[0] * kGravity,
                                 imuInput.a[1] * kGravity,
                                 imuInput.a[2] * kGravity,
                                 imuInput.g1[0],
                                 imuInput.g1[1],
                                 imuInput.g1[2]};
            meanValue.ax += imuValue.ax;
            meanValue.ay += imuValue.ay;
            meanValue.az += imuValue.az;
            meanValue.gx += imuValue.gx;
            meanValue.gy += imuValue.gy;
            meanValue.gz += imuValue.gz;
            imuValues.push_back(imuValue);
        }
    }

    meanValue.ax /= double(imuValues.size());
    meanValue.ay /= double(imuValues.size());
    meanValue.az /= double(imuValues.size());
    meanValue.gx /= double(imuValues.size());
    meanValue.gy /= double(imuValues.size());
    meanValue.gz /= double(imuValues.size());

    for (auto imuValue : imuValues)
    {
        meanValue.stdAx += (imuValue.ax - meanValue.ax) * (imuValue.ax - meanValue.ax);
        meanValue.stdAy += (imuValue.ay - meanValue.ay) * (imuValue.ay - meanValue.ay);
        meanValue.stdAz += (imuValue.az - meanValue.az) * (imuValue.az - meanValue.az);
        meanValue.stdGx += (imuValue.gx - meanValue.gx) * (imuValue.gx - meanValue.gx);
        meanValue.stdGy += (imuValue.gy - meanValue.gy) * (imuValue.gy - meanValue.gy);
        meanValue.stdGz += (imuValue.gz - meanValue.gz) * (imuValue.gz - meanValue.gz);
    }

    meanValue.stdAx /= double(imuValues.size() - 1);
    meanValue.stdAy /= double(imuValues.size() - 1);
    meanValue.stdAz /= double(imuValues.size() - 1);
    meanValue.stdGx /= double(imuValues.size() - 1);
    meanValue.stdGy /= double(imuValues.size() - 1);
    meanValue.stdGz /= double(imuValues.size() - 1);

    return meanValue;
}