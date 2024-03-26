# 模型转换

## 运行环境

```
git clone https://github.com/chenyangMl/keyword-spot.git
conda create -n kws python=3.9
conda activate kws
pip install -r requirements.txt

cd keyword-spot
mkdir models
```



下面示例模型转换，示例均使用“你好问问”数据训练的模型。

## Max-pooling方案模型转换

1. 先下载模型

   ```
   cd models
   
   下载方式1
   git clone https://www.modelscope.cn/daydream-factory/keyword-spot-dstcn-maxpooling-wenwen.git
   
   下载方式2
   from modelscope import snapshot_download
   model_dir = snapshot_download('daydream-factory/keyword-spot-dstcn-maxpooling-wenwen')
   ```

模型目录结构如下 >> tree keyword-spot-dstcn-maxpooling-wenwen

```
keyword-spot-dstcn-maxpooling-wenwen
├── avg_30.pt
├── configuration.json
├── config.yaml
├── global_cmvn
├── README.md
```



2 模型转换。

确定下当前在主目录路径，比如示例目录/path/keyword-spotting/models/

```
>> cd ../
>> pwd 
```

模型转换 1) pytorch to onnx

```
python model_convert/export_onnx.py \
 --config models/keyword-spot-dstcn-maxpooling-wenwen/config.yaml \
 --checkpoint models/keyword-spot-dstcn-maxpooling-wenwen/avg_30.pt \
 --onnx_model models/keyword-spot-dstcn-maxpooling-wenwen/onnx/keyword-spot-dstcn-maxpooling-wenwen.onnx
```



2) onnx2ort. 用于端侧设备部署.

```
python -m onnxruntime.tools.convert_onnx_models_to_ort models/keyword-spot-dstcn-maxpooling-wenwen/onnx/keyword-spot-dstcn-maxpooling-wenwen.onnx
```

3) 输出模型结构,>> tree models/keyword-spot-dstcn-maxpooling-wenwen

```
models/keyword-spot-dstcn-maxpooling-wenwen
├── avg_30.pt
├── configuration.json
├── config.yaml
├── global_cmvn
├── onnx
│   ├── keyword-spot-dstcn-maxpooling-wenwen.onnx  #中间模型
│   ├── keyword-spot-dstcn-maxpooling-wenwen.ort   #用于端侧部署的ort模型
│   ├── keyword-spot-dstcn-maxpooling-wenwen.required_operators.config
│   ├── keyword-spot-dstcn-maxpooling-wenwen.required_operators.with_runtime_opt.config
│   └── keyword-spot-dstcn-maxpooling-wenwen.with_runtime_opt.ort
├── README.md
└── words.txt
```



## CTC方案模型转换

1 下载模型

```
cd models

下载方式1
git clone https://www.modelscope.cn/daydream-factory/keyword-spot-fsmn-ctc-wenwen.git

下载方式2
from modelscope import snapshot_download
model_dir = snapshot_download('daydream-factory/keyword-spot-fsmn-ctc-wenwen')
```

模型目录查看>> tree keyword-spot-fsmn-ctc-wenwen/

```
keyword-spot-fsmn-ctc-wenwen/
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
cd path_to/keyword-spotting/
#在工程根目录运行
python model_convert/export_onnx.py \
 --config models/keyword-spot-fsmn-ctc-wenwen/config.yaml \
 --checkpoint models/keyword-spot-fsmn-ctc-wenwen/avg_30.pt \
 --onnx_model models/keyword-spot-fsmn-ctc-wenwen/onnx/keyword_spot_fsmn_ctc_wenwen.onnx
```



2) onnx2ort. 用于端侧设备部署.

```
python -m onnxruntime.tools.convert_onnx_models_to_ort models/keyword-spot-fsmn-ctc-wenwen/onnx/keyword_spot_fsmn_ctc_wenwen.onnx
```

3) 输出模型结构

```
models/keyword-spot-fsmn-ctc-wenwen
├── avg_30.pt
├── configuration.json
├── config.yaml
├── global_cmvn.kaldi
├── lexicon.txt
├── onnx
│   ├── keyword_spot_fsmn_ctc_wenwen.onnx #中间模型
│   ├── keyword_spot_fsmn_ctc_wenwen.ort  #用于端侧部署的ort模型
│   ├── keyword_spot_fsmn_ctc_wenwen.required_operators.config
│   ├── keyword_spot_fsmn_ctc_wenwen.required_operators.with_runtime_opt.config
│   └── keyword_spot_fsmn_ctc_wenwen.with_runtime_opt.ort
├── README.md
└── tokens.txt
```



1. 模型下载

```
git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone-wenwen.git

```



## 模型可视化工具netron

使用模型可视化工具可以方便查看模型的整体结构，输入输出信息等，便于校验转换模型。

```
pip install netron
```

查看模型使用命令

```
netron path_to_model
```

打开提供的链接即可浏览器查看。