set(ONNX_VERSION "1.19.2")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-aarch64-${ONNX_VERSION}.tgz")
  set(URL_HASH "SHA256=5e30145277d6d6fcb0e8f14f0d0ab5048af7b13ffd608023bb1e2875621fab07")
else()
  set(ONNX_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-${ONNX_VERSION}.tgz")
  set(URL_HASH "SHA256=eb00c64e0041f719913c4080e0fed7d9963dc3aa9b54664df6036d8308dbcd33")
endif()

FetchContent_Declare(onnxruntime
  URL ${ONNX_URL}
  URL_HASH ${URL_HASH}
)
FetchContent_MakeAvailable(onnxruntime)
include_directories(${onnxruntime_SOURCE_DIR}/include)
link_directories(${onnxruntime_SOURCE_DIR}/lib)
