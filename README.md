# 关键词检测

本工程主要基于[wekws](https://github.com/wenet-e2e/wekws/tree/main)进行构建，旨在搭建基于当下新训练框架(HF, modelscope)来实现更高效、快速的模型训练，微调及部署落地。



# Features

- 支持CTC和Max-Pooling方案的唤醒词模型推理。
- 支持模型转换，Pytorch2ONNX,  ONNX2ORT(端侧部署)
- 支持CPP onnxruntime流式推理.



# Change Log

- 2024/03/21 : 提供完整的CTC和Max-Pooling唤醒词方案的onnx模型cpp推理测试。
- 2024/03/26:  提供模型转换工具，支持模型从Pytorch转换到onnx,再转换到ort用于端侧部署。支持CPP onnxruntime流式推理.





# 推理测试

以下示例 CPP  ONNX推理测试.

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
./kws_main [solution_type, int] [num_bins, int] [batch_size, int] [model_path, str] [wave_path,str]

#eg
./kws_main 0 40 1 keyword-spot-dstcn-maxpooling-wenwen/onnx/keyword-spot-dstcn-maxpooling-wenwen.ort ../../../audio/0000c7286ebc7edef1c505b78d5ed1a3.wav
```



## CTC 方案模型

```
cd build/bin
./kws_main [solution_type, int] [num_bins, int] [batch_size, int] [model_path, str] [wave_path,str] [key_word,str]

#eg
./kws_main 1 80 1 keyword_spot_fsmn_ctc_wenwen/onnx/keyword_spot_fsmn_ctc_wenwen.ort ../../../audio/0000c7286ebc7edef1c505b78d5ed1a3.wav 你好小问
```

更多详细信息参考: [onnx runtime](onnxruntime/README.md)

PS: solution_type:{0:表示max-pooling方案, 1:表示ctc方案}

​     key_word: {你好小问，嗨小问}



如需要其他端测的推理测试，可参考wekws提供的[Android, RaspberryPI示例](https://github.com/wenet-e2e/wekws/tree/main/runtime)。



# 模型训练

xxx





# 模型转换

- pytorch2onnx: 将训练好的pytorch模型转换为onnx模型。onnx模型是常见的中间态模型，支持转换其他平台的模型(ncnn, tensorRT等各类推理引擎模型)。
- onnx2ort: 将onnx模型转换成ort模型，用于端侧部署。

详细内容参考[唤醒词模型转换](docs/model_convert.md)





## 模型列表

| 损失函数    | 模型名称         | 模型(Pytorch ckpt)                                           | 模型(ONNX)                                                   | 端侧模型                                                     |
| ----------- | ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Max-Pooling | DS_TCN(你好问问) | [DSTCN-MaxPooling, wekws训练](https://modelscope.cn/models/daydream-factory/keyword-spot-dstcn-maxpooling-wenwen/summary) | [ONNX](https://modelscope.cn/models/daydream-factory/keyword-spot-dstcn-maxpooling-wenwen/files) | [ORT](https://modelscope.cn/models/daydream-factory/keyword-spot-dstcn-maxpooling-wenwen/files) |
|             |                  |                                                              |                                                              |                                                              |
| CTC         | FSMN(你好问问)   | [FSMN-CTC, wekws训练](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-wenwen/summary) | [ONNX](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-wenwen/files) | [ORT](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-wenwen/files) |
|             |                  |                                                              |                                                              |                                                              |
| CTC         | FSMN(你好问问)   | [FSMN-CTC, modelscope训练](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-nihaowenwen/summary) | [ONNX](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-nihaowenwen/files) | [ORT](https://modelscope.cn/models/daydream-factory/keyword-spot-fsmn-ctc-nihaowenwen/files) |







## 参考＆鸣谢

  本工程主要基于[wekws](https://github.com/wenet-e2e/wekws/tree/main)进行语音唤醒的模型训练，模型转换，推理，部署等流程构建，特此感谢。

- [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)  : CTC的设计思路和原理。
- [魔搭: 你好问问 唤醒词检测体验测试Demo](https://modelscope.cn/studios/thuduj12/KWS_Nihao_Xiaojing/summary)
- https://modelscope.cn/models/iic/speech_charctc_kws_phone-wenwen/summary



# [License](./LICENSE)

MIT
