# Consumer smoke test: a separate project that finds an installed OpenNN
# with find_package and links a one-file program against it. It is not part
# of the main build; CI installs the library to a prefix and configures this
# directory against it:
#
#   cmake --install <build> --prefix <prefix>
#   cmake -S tools/package_smoke -B <build>/package -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build <build>/package && <build>/package/opennn_package_smoke
cmake_minimum_required(VERSION 3.24)
project(OpenNNPackageSmoke LANGUAGES CXX)

find_package(OpenNN 8.0 CONFIG REQUIRED)

add_executable(opennn_package_smoke main.cpp)
target_link_libraries(opennn_package_smoke PRIVATE OpenNN::opennn)
target_compile_features(opennn_package_smoke PRIVATE cxx_std_20)
