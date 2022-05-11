cd procgen/
cmake -B jsbuild -DCMAKE_TOOLCHAIN_FILE=/opt/cheerp/share/cmake/Modules/CheerpToolchain.cmake .
cmake --build jsbuild 
cmake --install jsbuild