cmake_minimum_required(VERSION 3.16)
project(VAD VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Position Independent Code for all targets
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------
# Build Options
# -------------------------------------------------------
option(BUILD_VAD_EXAMPLES "Build VAD example programs" ON)
option(BUILD_VAD_PYTHON "Build Python bindings" ON)

# -------------------------------------------------------
# Find Dependencies
# -------------------------------------------------------
find_package(PkgConfig REQUIRED)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime
    /opt/homebrew/include/onnxruntime
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS
    /usr/local/lib
    /usr/lib
    /opt/homebrew/lib
)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR
        "ONNX Runtime not found!\n"
        "Install ONNX Runtime:\n"
        "  macOS: brew install onnxruntime\n"
        "  Linux: Download from https://github.com/microsoft/onnxruntime/releases"
    )
endif()

# libcurl (for model auto-download)
find_package(CURL REQUIRED)

# -------------------------------------------------------
# Silero Backend Library
# -------------------------------------------------------
set(SILERO_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backends/silero/silero_backend.cpp
)

add_library(silero_vad STATIC ${SILERO_SOURCES})

target_include_directories(silero_vad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/internal
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(silero_vad
    PUBLIC ${ONNXRUNTIME_LIB}
    PRIVATE CURL::libcurl pthread
)

set_target_properties(silero_vad PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# -------------------------------------------------------
# VAD Framework Library
# -------------------------------------------------------
set(VAD_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vad_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/vad_backend_factory.cpp
)

add_library(vad STATIC ${VAD_SOURCES})

target_include_directories(vad PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc/internal
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(vad
    PUBLIC silero_vad
    PRIVATE pthread
)

set_target_properties(vad PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

message(STATUS "VAD Framework library enabled")

# -------------------------------------------------------
# VAD Examples (optional)
# -------------------------------------------------------
if(BUILD_VAD_EXAMPLES)
    # Simple demo
    add_executable(vad_simple_demo ${CMAKE_CURRENT_SOURCE_DIR}/examples/simple_demo.cpp)

    target_include_directories(vad_simple_demo PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
    )

    target_link_libraries(vad_simple_demo
        PRIVATE vad
        PRIVATE pthread
        PRIVATE m
    )

    set_target_properties(vad_simple_demo PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # Stream demo (if audio component and source file available)
    set(AUDIO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../audio)
    set(STREAM_DEMO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/examples/stream_demo.cpp)
    if(EXISTS ${AUDIO_DIR} AND EXISTS ${STREAM_DEMO_SRC})
        pkg_check_modules(PORTAUDIO portaudio-2.0)
        if(PORTAUDIO_FOUND)
            add_executable(vad_stream_demo ${STREAM_DEMO_SRC})

            target_include_directories(vad_stream_demo PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/inc
                ${AUDIO_DIR}/inc
                ${PORTAUDIO_INCLUDE_DIRS}
            )

            target_link_libraries(vad_stream_demo
                PRIVATE vad
                PRIVATE ${PORTAUDIO_LIBRARIES}
                PRIVATE pthread
                PRIVATE m
            )

            target_link_directories(vad_stream_demo PRIVATE ${PORTAUDIO_LIBRARY_DIRS})

            set_target_properties(vad_stream_demo PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
            )

            message(STATUS "VAD Stream Demo enabled")
        endif()
    endif()

    message(STATUS "VAD Examples enabled")
else()
    message(STATUS "VAD Examples disabled (BUILD_VAD_EXAMPLES=OFF)")
endif()

# -------------------------------------------------------
# Python Bindings (optional)
# -------------------------------------------------------
if(BUILD_VAD_PYTHON)
    set(VAD_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/python/vad_bindings.cpp)
    if(NOT EXISTS ${VAD_BINDINGS_SRC})
        message(STATUS "Python bindings disabled (vad_bindings.cpp not found)")
        set(BUILD_VAD_PYTHON OFF)
    endif()
endif()

if(BUILD_VAD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)

    if(Python3_FOUND)
        message(STATUS "Python found: ${Python3_EXECUTABLE} (${Python3_VERSION})")
    endif()

    find_package(pybind11 CONFIG QUIET)

    if(NOT pybind11_FOUND AND Python3_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE PYBIND11_RESULT
        )
        if(PYBIND11_RESULT EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG QUIET)
        endif()
    endif()

    if(pybind11_FOUND)
        pybind11_add_module(_evo_vad ${CMAKE_CURRENT_SOURCE_DIR}/python/vad_bindings.cpp)

        target_include_directories(_evo_vad PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/inc
        )

        target_link_libraries(_evo_vad PRIVATE vad)

        # Only set default output dir if not specified externally (e.g., by setup.py)
        if(NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
            set_target_properties(_evo_vad PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
            )
        endif()

        message(STATUS "Python bindings enabled (_evo_vad module for Python ${Python3_VERSION})")

        # Get Python site-packages path
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_path('purelib'))"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        # Install target
        add_custom_target(vad-install-python
            COMMAND ${CMAKE_COMMAND} -E remove_directory ${PYTHON_SITE_PACKAGES}/evo_vad
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/evo_vad ${PYTHON_SITE_PACKAGES}/evo_vad
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_evo_vad> ${PYTHON_SITE_PACKAGES}/evo_vad/
            DEPENDS _evo_vad
            COMMENT "Installing evo_vad to ${PYTHON_SITE_PACKAGES}/evo_vad/"
        )
    else()
        message(STATUS "Python bindings disabled (pybind11 not found)")
    endif()
endif()

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
message(STATUS "")
message(STATUS "=== VAD Build Configuration ===")
message(STATUS "  BUILD_VAD_EXAMPLES:  ${BUILD_VAD_EXAMPLES}")
message(STATUS "  BUILD_VAD_PYTHON:    ${BUILD_VAD_PYTHON}")
message(STATUS "  ONNX Runtime:        ${ONNXRUNTIME_LIB}")
message(STATUS "")
