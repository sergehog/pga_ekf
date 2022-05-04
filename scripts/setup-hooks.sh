#!/usr/bin/env bash

GIT_DIR=$(git rev-parse --git-dir)

cd "${0%/*}/.."
CURR_DIR=$(pwd)

echo "Installing GIT hooks..."
# this command creates symlink to pre-push script
ln -s $CURR_DIR/scripts/pre-push.sh $GIT_DIR/hooks/pre-push


