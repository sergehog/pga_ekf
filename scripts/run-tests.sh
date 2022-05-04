#!/usr/bin/env bash

# if any command inside script returns error, exit and return that error 
set -e

cd "${0%/*}/.."
CURR_DIR=$(pwd)

mkdir -p /tmp/tiny_pga_tests
cd /tmp/tiny_pga_tests
cmake -DCMAKE_BUILD_TYPE=Release $CURR_DIR
make
ctest


