cmake_minimum_required(VERSION 3.22)
project(Backend2 VERSION 0.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_PREFIX_PATH "/home/maksuning/Qt/6.8.1/gcc_64")

find_package(Qt6 REQUIRED COMPONENTS Core Network HttpServer)

set(SOURCES
    src/main.cpp
    src/Com/server.cpp
    src/engine/physics.cpp
    src/engine/session_mgr.cpp
)

set(HEADERS
    src/config.h
    src/Com/server.h
    src/Com/handlers/health_handler.h
    src/Com/handlers/energy_calc_handler.h
    src/Com/handlers/predict_math_handler.h
    src/Com/handlers/session_handler.h
    src/engine/physics.h
    src/engine/session_mgr.h
)

add_executable(backend2 ${SOURCES} ${HEADERS})

target_include_directories(backend2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(backend2 PRIVATE
    Qt6::Core
    Qt6::Network
    Qt6::HttpServer
    dl
)

set_target_properties(backend2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
