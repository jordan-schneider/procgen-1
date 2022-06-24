#!/bin/bash
./procgen/build.sh
cp procgen/jsbuild/dist/env.js ~/experiment-server/experiment_server/static/
cp -r procgen/jsbuild/dist/assets/* ~/experiment-server/experiment_server/static/