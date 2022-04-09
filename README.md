# pga_ekf : Extended Kalman Filter using PGA Motor algebra 

**Plane-based** (or **Projective**) **Geometric Algebra** (PGA) in 3D is powerful tool for solving various kinds of Comuter Geometry problems.
This project is attempt to solve classical IMU Dead Reckoning problem using PGA Motor algebra.

Apparently, power and flexibility of PGA are so high, that adding other types of sensors dosn't brings any problems.
At this moment we assume that our setup contains 6DoF IMU-sensor as well as some external 3S position sensor, such as GPS.

High-level implementation details hidden by optimization, thus we only see bunch of equations.  