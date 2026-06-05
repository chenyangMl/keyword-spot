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





## 自定义CTC模型转换（基于modelscope xiaoyun微调）

当你在 `speech_charctc_kws_phone-xiaoyun` 基础上训练/微调得到自定义CTC模型后，
直接用上面的命令转换可能会报错（参考 issue #5 / #15）：

```
RuntimeError: Error(s) in loading state_dict for KWSModel:
        Missing key(s) in state_dict: "classifier.linear.weight", "classifier.linear.bias".
        Unexpected key(s) in state_dict: "global_cmvn.mean", "global_cmvn.istd".
```

原因是转换用的 `config.yaml` 与 checkpoint 的网络结构不一致：

- **`Unexpected key: global_cmvn.*`**：该 checkpoint 把全局 cmvn 统计量直接保存在了
  `.pt` 文件里（`global_cmvn.mean/istd`），但你的 config 没有构建 cmvn 模块。
- **`Missing key: classifier.linear.*`**：CTC/FSMN 模型的 backbone 已经直接输出
  `output_dim`，分类器应当是 `identity`，而你的 config 构建了带权重的 `LinearClassifier`。

修复方法：在转换用的 `config.yaml` 的 `model` 段中做如下两处调整。

```yaml
model:
  input_dim: 80
  output_dim: <你的token数量>   # 与 tokens.txt 行数一致
  hidden_dim: <与训练一致>
  # 1) 让分类器为 identity（CTC/FSMN backbone 已输出 output_dim）
  classifier:
    type: identity
    dropout: 0.1
  # 2) 复用 checkpoint 中保存的 global_cmvn（无需单独的 cmvn 文件）
  cmvn:
    load_from_checkpoint: true
    norm_var: true
  backbone:
    type: fsmn
    # ... 其余 fsmn 参数与训练时保持一致 ...
```

调整后再执行 `python model_convert/export_onnx.py ...` 即可正常转换。
若仍有 key 不匹配，`load_checkpoint` 会打印出具体缺失/多余的 key 及修复建议。


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