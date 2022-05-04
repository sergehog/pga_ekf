#!/usr/bin/env bash

cd "${0%/*}/.."

find . -regex '.*\.\(c\|cpp\|cc\|cxx\|h\|hpp\)' -exec clang-format -style=file -i {} \;