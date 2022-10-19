#!/usr/bin/env bash

cd "${0%/*}/.."

find . -regex '.*\.\(c\|cpp\|cc\|cxx\|h\|hpp\)' -not -path "./samples/openzen/openzen/*" -not -path "./*build*/*" -exec clang-format -style=file -i {} \;