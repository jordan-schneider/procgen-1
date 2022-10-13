#!/bin/bash
./procgen/build.sh
cp procgen/jsbuild/dist/env.js ~/research/experiment-server/experiment_server/static/
cp -r procgen/jsbuild/dist/assets/* ~/research/experiment-server/experiment_server/static/