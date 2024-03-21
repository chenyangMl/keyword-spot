# 关键词检测





# 推理测试

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

更多详细信息参考: [onnx runtime](onnxruntime/README.md)





# 模型训练

xxx





# 模型转换

xxx





## 模型列表

| 模型名称 | 损失函数(Loss) | 模型(Pytorch ckpt) | 模型(onnx) | 模型() |
| -------- | -------------- | ------------------ | ---------- | ------ |
|          |                |                    |            |        |
|          |                |                    |            |        |
|          |                |                    |            |        |











## 参考＆鸣谢

本工程主要是基于[wekws](https://github.com/wenet-e2e/wekws/tree/main)进行构建的，特此感谢。

- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)  : CTC的设计思路和原理。
- [你好问问唤醒词检测](https://modelscope.cn/studios/thuduj12/KWS_Nihao_Xiaojing/summary)

