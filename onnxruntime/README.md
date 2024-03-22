

# CPP  ONNX推理测试.



```shell
git clone https://github.com/chenyangMl/keyword-spot.git
cd keyword-spot/onnxruntime/
mkdir build && cd build 
cmake .. 
cmake --build . --target kws_main  

#不同模型使用如下对应参数进行模型推理。
```



## [测试音频](../audio)

| 音频名称                                | 关键字         |
| --------------------------------------- | -------------- |
| 000af5671fdbaa3e55c5e2bd0bdf8cdd_hi.wav | 嗨小问         |
| 000eae543947c70feb9401f82da03dcf_hi.wav | 嗨小问         |
| 0000c7286ebc7edef1c505b78d5ed1a3.wav    | 你好小问       |
| 0000e12e2402775c2d506d77b6dbb411.wav    | 你好小问       |
| gongqu-4.5_0000.wav                     | 其他(测试负例) |



## Max-Pooling方案模型

```
cd build/bin
./kws_main 0 40 1 path_to_your_model.ort path_to_your_wave.wav

#eg
./kws_main 0 40 1 keyword-spot-dstcn-maxpooling-wenwen/onnx/keyword-spot-dstcn-maxpooling-wenwen.ort ../../../audio/0000c7286ebc7edef1c505b78d5ed1a3.wav
```

测试日志:  frame表示当前处理的time step. prob的第一列表示关键词1的分类概率, 　第二列关键词2的分类概率。

```
> Kws Model Info:
> 	cache_dim: 256
> 	cache_len: 105
> frame 0 prob 4.17233e-07 1.49012e-06
> frame 1 prob 3.8743e-07 1.2517e-06
> frame 2 prob 1.19209e-07 5.36442e-07
> frame 3 prob 2.98023e-07 3.01003e-06
>
> ...
>
> frame 100 prob 0.963686 activated keyword: 嗨小问  0
> frame 101 prob 0.955697 activated keyword: 嗨小问  0
> frame 102 prob 0.94719 activated keyword: 嗨小问  0
> frame 103 prob 0.909599 activated keyword: 嗨小问  2.98023e-08
> frame 104 prob 0.985421 activated keyword: 嗨小问  2.98023e-08
> frame 105 prob 0.926912 activated keyword: 嗨小问  2.98023e-08
> frame 106 prob 0.980361 activated keyword: 嗨小问  0
> frame 107 prob 0.988708 activated keyword: 嗨小问  0
> frame 108 prob 0.998589 activated keyword: 嗨小问  0
>
> ...
>
> frame 149 prob 2.98023e-08 0
> frame 150 prob 8.9407e-08 0
> frame 151 prob 1.19209e-07 2.98023e-08
> frame 152 prob 0 0
> frame 153 prob 2.98023e-08 0
> frame 154 prob 2.98023e-08 0
>
> Process finished with exit code 0
```



## CTC 方案模型

```
cd build/bin
./kws_main 0 80 1 path_to_your_model.ort path_to_your_wave.wav [keyword]

#eg
./kws_main 1 80 1 keyword_spot_fsmn_ctc_wenwen/onnx/keyword_spot_fsmn_ctc_wenwen.ort ../../../audio/0000c7286ebc7edef1c505b78d5ed1a3.wav 你好小问
```

测试日志: 如下是CTC prefix beam search的

 frame表示当前处理的time step. tokenid:表示当前帧识别到的Token ID.  proposed:表示基于当前假设(current hypotheses) 的扩展(proposed extensions). 建议参考图示[Sequence Modeling With CTC](https://distill.pub/2017/ctc/) 理解。prob表示该token的分类概率。



```
Kws Model Info:
	cache_dim: 128
	cache_len: 11
stepT=  0 tokenid=   0 proposed i=0 prob=0.952
stepT=  3 tokenid=   0 proposed i=0 prob=0.943
stepT=  6 tokenid=   0 proposed i=0 prob=0.946
stepT=  9 tokenid=   0 proposed i=0 prob=0.965
stepT= 12 tokenid=   0 proposed i=0 prob=0.801

...

stepT=129 tokenid=   0 proposed i=0 prob=1
stepT=132 tokenid=2494 proposed i=0 prob=0.954
hitword=你好小问
hitscore=0.954
start frame=69 end frame=132
stepT=135 tokenid=2494 proposed i=0 prob=1
stepT=138 tokenid=   0 proposed i=0 prob=1
stepT=141 tokenid=   0 proposed i=0 prob=1
stepT=144 tokenid=   0 proposed i=0 prob=1
stepT=147 tokenid=   0 proposed i=0 prob=1
stepT=150 tokenid=   0 proposed i=0 prob=1
stepT=153 tokenid=   0 proposed i=0 prob=1
stepT=156 tokenid=   0 proposed i=0 prob=1
stepT=159 tokenid=   0 proposed i=0 prob=1
stepT=162 tokenid=   0 proposed i=0 prob=1
stepT=165 tokenid=   0 proposed i=0 prob=1

Process finished with exit code 0
```

