
find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem)
include_directories(untitled ${Boost_INCLUDE_DIRS})

add_executable(kws_main kws_main.cc)
target_link_libraries(kws_main PUBLIC onnxruntime frontend kws ${Boost_LIBRARIES})

add_executable(kws_testing kws_testing.cc)
target_link_libraries(kws_testing PUBLIC onnxruntime frontend kws ${Boost_LIBRARIES})

add_executable(device_test device_test.cc)
target_link_libraries(device_test PUBLIC portaudio_static)

add_executable(stream_kws_main stream_kws_main.cc)
target_link_libraries(stream_kws_main PUBLIC onnxruntime frontend kws portaudio_static)