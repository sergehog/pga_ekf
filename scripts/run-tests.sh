#!/usr/bin/env bash

# if any command inside script returns error, exit and return that error 
set -e

cd "${0%/*}/.."
CURR_DIR=$(pwd)

mkdir -p /tmp/pga_ekf_tests
cd /tmp/pga_ekf_tests
cmake -DCMAKE_BUILD_TYPE=Release $CURR_DIR
make -j4
ctest


