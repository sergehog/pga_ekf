#!/usr/bin/env bash

cd "${0%/*}/.."

find . -regex '.*\.\(c\|cpp\|cc\|cxx\|h\|hpp\)' -not -path "./openzen/*" -not -path "./*build*/*" -exec clang-tidy {} -checks=\'*\' \;