#!/bin/bash
./procgen/build.sh
cp procgen/jsbuild/dist/env.js experiment-server/experiment_server/static/
cp -r procgen/jsbuild/dist/assets/* experiment-server/experiment_server/static/
# Realistically I need multiple pages, there's nothing super interesting in this html file, it just
# gets copied over a bunch.
# cp procgen/jsbuild/dist/index.html experiment-server/experiment_server/templates/