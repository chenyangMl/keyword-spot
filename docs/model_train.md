 ## 语言唤醒——模型训练

工程示例提供了基于问问开源的唤醒词模型，支持{你好小问，嗨小问}两个唤醒词。

如果需要定制新的唤醒词可模型微调。

- 先确定好唤醒词，可以是单唤醒词，也可以是多唤醒词。比如{你好小问，嗨小问}

- 采集或模拟真实应用场景的唤醒词数据。　
- 进行模型微调得到新的唤醒词模型。



## [max-pooling方案训练](https://zhuanlan.zhihu.com/p/686365901)





## CTC方案训练

### 数据采集

训练数据需包含一定数量对应关键词(正例)和非关键词(负例)样本。

**[达摩小云小云模型训练](https://www.modelscope.cn/models/iic/speech_charctc_kws_phone-xiaoyun/summary)建议关键词数据在25小时以上，混合负样本比例在1:2到1:10之间，实际性能与训练数据量、数据质量、场景匹配度、正负样本比例等诸多因素有关，需要具体分析和调整**。



```
示例数据目录结构
unittest/example_kws
└── wav #wav格式的音频数据
	├── 1330806238146100615.wav
    ├── 20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_10.wav
    ├── 20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_11.wav
    ├── 20200707_spk57db_storenoise52db_40cm_xiaoyun_sox_12.wav
├── cv_wav.scp    #验证集
├── test_wav.scp  #测试集
├── train_wav.scp #训练集
├── merge_trans.txt

#示例文件内容
$ cat cv_wav.scp
kws_pos_example1	/home/admin/data/test/audios/kws_pos_example1.wav
kws_pos_example2	/home/admin/data/test/audios/kws_pos_example2.wav
...
kws_neg_example1	/home/admin/data/test/audios/kws_neg_example1.wav
kws_neg_example2	/home/admin/data/test/audios/kws_neg_example2.wav
...

$ cat merge_trans.txt
kws_pos_example1	小 云 小 云
kws_pos_example2	小 云 小 云
...
kws_neg_example1	帮 我 导航 一下 回 临江 路 一百零八 还要 几个 小时
kws_neg_example2	明天 的 天气 怎么样
...
```



### 模型训练

模型训练采用"basetrain + finetune"的模式，**basetrain过程使用大量内部移动端数据**，在此基础上，使用1万条设备端录制安静场景“小云小云”数据进行微调，得到最终面向业务的模型。由于采用了中文char全量token建模，并使用充分数据进行basetrain，本模型支持基本的唤醒词/命令词自定义功能，但具体性能无法评估。



```
#下载训练工具
git clone https://www.modelscope.cn/iic/speech_charctc_kws_phone-xiaoyun.git
cd unittest/

CUDA_VISIBLE_DEVICES=0 python example_kws.py
```



