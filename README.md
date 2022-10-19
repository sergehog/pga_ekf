![CI Status](https://github.com/sergehog/pga_ekf/actions/workflows/master.yml/badge.svg?branch=master)

# pga_ekf : Extended Kalman Filter using PGA Motor algebra 

**Plane-based** (aka **Projective**) **Geometric Algebra** (PGA) in 3D is powerful tool for solving various kinds of Comuter Geometry problems.
This project is attempt to solve classical **IMU+GPS Dead Reckoning** problem using PGA Motor algebra.


At this moment we assume that our setup contains **6DoF IMU sensor** as well as some external **3D position sensor**, such as GPS.
Power and flexibility of PGA are so high, that adding other types of sensors doesn't bring significant hurdle.

High-level implementation details and explanations are hidden by purpose. Please contact me if you want to see them, for instance for commercial or other kind of use.   
