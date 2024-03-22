# 关键词检测

本工程主要基于[wekws](https://github.com/wenet-e2e/wekws/tree/main)进行构建，旨在搭建基于当下新训练框架(HF, modelscope)来实现更高效、快速的模型训练，微调及部署落地。



# Features

- 支持CTC和Max-Pooling方案的唤醒词模型推理。



# Change Log

- 2024/03/21 : 提供完整的CTC和Max-Pooling唤醒词方案的onnx模型cpp推理测试。





# 推理测试

 CPP  ONNX推理测试.

```shell
git clone https://github.com/chenyangMl/keyword-spot.git
cd keyword-spot/onnxruntime/
mkdir build && cd build 
cmake .. 
cmake --build . --target kws_main  

#不同模型使用如下对应参数进行模型推理。
```



## Max-Pooling方案模型

```
cd build/bin
./kws_main 0 40 1 path_to_your_model.ort path_to_your_wave.wav

#eg
./kws_main 0 40 1 keyword-spot-dstcn-maxpooling-wenwen/onnx/keyword-spot-dstcn-maxpooling-wenwen.ort ../../../audio/0000c7286ebc7edef1c505b78d5ed1a3.wav
```



## CTC 方案模型

```
cd build/bin
./kws_main 0 80 1 path_to_your_model.ort path_to_your_wave.wav [keyword]

#eg
./kws_main 1 80 1 keyword_spot_fsmn_ctc_wenwen/onnx/keyword_spot_fsmn_ctc_wenwen.ort ../../../audio/0000c7286ebc7edef1c505b78d5ed1a3.wav 你好小问
```

更多详细信息参考: [onnx runtime](onnxruntime/README.md)





# 模型训练

xxx





# 模型转换

- pytorch2onnx: 将训练好的pytorch模型转换为onnx模型。onnx模型是常见的中间态模型，支持转换其他平台的模型(ncnn, tensorRT等各类推理引擎模型)。
- onnx2ort: 将onnx模型转换成ort模型，用于端侧部署。

详细内容参考[唤醒词模型转换](docs/model_convert.md)





## 模型列表

| 损失函数    | 模型名称         | 模型(Pytorch ckpt)                                           | 模型(ONNX)                                                   | 端侧模型                                                     |
| ----------- | ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Max-Pooling | DS_TCN(你好问问) | [DSTCN-MaxPooling](https://modelscope.cn/models/daydream-factory/keyword-spot-dstcn-maxpooling-wenwen/files) | [ONNX](https://modelscope.cn/models/daydream-factory/keyword-spot-dstcn-maxpooling-wenwen/files) | [ORT](https://modelscope.cn/models/daydream-factory/keyword-spot-dstcn-maxpooling-wenwen/files) |
|             |                  |                                                              |                                                              |                                                              |
| CTC         | FSMN(你好问问)   | [FSMN-CTC](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-wenwen/summar) | [ONNX](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-wenwen/files) | [ORT](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-wenwen/files) |







## 参考＆鸣谢

  本工程主要是基于[wekws](https://github.com/wenet-e2e/wekws/tree/main)进行构建的，特此感谢。

- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)  : CTC的设计思路和原理。
- [魔搭: 你好问问 唤醒词检测体验测试Demo](https://modelscope.cn/studios/thuduj12/KWS_Nihao_Xiaojing/summary)
- https://modelscope.cn/models/iic/speech_charctc_kws_phone-wenwen/summary



# [License](./LICENSE)

MIT

