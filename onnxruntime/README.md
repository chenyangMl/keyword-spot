CPP  ONNX推理测试.

```shell
git clone 

mkdir build && cd build 
cmake .. 
cmake --build . --target kws_main  
```



## Max-Pooling方案模型

```
./kws_main 0 40 1 path_to_your_model.ort path_to_your_wave.wav
```



## CTC 方案模型

```
./kws_main 0 80 1 path_to_your_model.ort path_to_your_wave.wav
```

