# 模型转换

## 运行环境

```
git clone https://github.com/wenet-e2e/wekws.git
conda create -n kws python=3.9
conda activate kws
pip install -r requirements.txt

cd wekws
mkdir models
```



## 模型查看工具netron

```
pip install netron
```

查看模型使用命令

```
netron path_to_model
```

打开提供的链接即可浏览器查看。



以下测试使用“你好问问”数据训练的模型，进行流程测试。

## Max-pooling方案模型转换

1. 先下载模型

   ```
   cd models
   
   下载方式1
   git clone https://www.modelscope.cn/thuduj12/kws_wenwen_dstcn.git
   
   下载方式2
   from modelscope import snapshot_download
   model_dir = snapshot_download('thuduj12/kws_wenwen_dstcn')
   ```

模型目录结构如下 >> tree kws_wenwen_dstcn. 

```
models/kws_wenwen_dstcn
├── avg_30.pt  #文件大小
├── configuration.json
├── config.yaml
├── global_cmvn
├── README.md
```



2 模型转换。

确定下当前在主目录路径，比如示例目录 /works/wekws.

```
>> cd ../
>> pwd 
```

模型转换 1) pytorch to onnx

```
python wekws/bin/export_onnx.py \
 --config models/kws_wenwen_dstcn/config.yaml \
 --checkpoint models/kws_wenwen_dstcn/avg_30.pt \
 --onnx_model models/kws_wenwen_dstcn/kws_wenwen_dstcn.onnx
```

> 可能遇到cmvn文件查找不到的情况。
>
> PS: FileNotFoundError: [Errno 2] No such file or directory: 'data/global_cmvn'
>
> 编辑器打开models/kws_wenwen_dstcn/config.yaml，将路径修改为基于工程的相对路径即可。比如这里的
>
> models/kws_wenwen_dstcn/global_cmvn



2) onnx2ort. 用于端侧设备部署.

```
python -m onnxruntime.tools.convert_onnx_models_to_ort models/kws_wenwen_dstcn/kws_wenwen_dstcn.onnx
```

3) 输出模型结构

```
models/kws_wenwen_dstcn
├── avg_30.pt
├── configuration.json
├── config.yaml
├── global_cmvn
├── kws_wenwen_dstcn.onnx #中间模型。
├── kws_wenwen_dstcn.ort #用于端侧部署的ort模型
├── kws_wenwen_dstcn.required_operators.config
├── kws_wenwen_dstcn.required_operators.with_runtime_opt.config
├── kws_wenwen_dstcn.with_runtime_opt.ort
├── README.md
└── words.txt
```



## CTC方案模型转换

1 下载模型

```
cd models

下载方式1
git clone https://www.modelscope.cn/thuduj12/kws_wenwen_fsmn_ctc.git

下载方式2
from modelscope import snapshot_download
model_dir = snapshot_download('thuduj12/kws_wenwen_fsmn_ctc')
```

模型目录查看>> tree kws_wenwen_fsmn_ctc

```
kws_wenwen_fsmn_ctc/
├── avg_30.pt
├── configuration.json
├── config.yaml
├── global_cmvn.kaldi
├── lexicon.txt
├── README.md
└── tokens.txt
```



2 模型转换

模型转换 1) pytorch to onnx

```
python wekws/bin/export_onnx.py \
 --config models/kws_wenwen_fsmn_ctc/config.yaml \
 --checkpoint models/kws_wenwen_fsmn_ctc/avg_30.pt \
 --onnx_model models/kws_wenwen_fsmn_ctc/kws_wenwen_fsmn_ctc.onnx
```

可能遇到cmvn文件查找不到的情况。

> PS: FileNotFoundError: [Errno 2] No such file or directory: 'data/global_cmvn.kaldi'
>
> 编辑器打开models/kws_wenwen_dstcn/config.yaml，将路径修改为基于工程的相对路径即可。比如这里的
>
> models/kws_wenwen_fsmn_ctc/global_cmvn.kaldi

2) onnx2ort. 用于端侧设备部署.

```
python -m onnxruntime.tools.convert_onnx_models_to_ort models/kws_wenwen_dstcn/kws_wenwen_dstcn.onnx
```

3) 输出模型结构

```
models/kws_wenwen_dstcn
├── avg_30.pt
├── configuration.json
├── config.yaml
├── global_cmvn
├── kws_wenwen_dstcn.onnx #onnx模型
├── kws_wenwen_dstcn.ort　#用于端侧部署的ort模型
├── kws_wenwen_dstcn.required_operators.config
├── kws_wenwen_dstcn.required_operators.with_runtime_opt.config
├── kws_wenwen_dstcn.with_runtime_opt.ort
├── README.md
└── words.txt
```