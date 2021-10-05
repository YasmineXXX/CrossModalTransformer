# Cross Modal Transformer

to prepare dataset:

[Perceptual Reasoning and Interaction Research - Charades (allenai.org)](https://prior.allenai.org/projects/charades)

to download the pretrained models:

链接：https://pan.baidu.com/s/1MQGJ3TgI-DrBvjPgD5E1MQ 
提取码：4kdm 

to train the model:

```
python train.py ${frame_dir} ${annotation_file}
```

to test the model:

```
python inference.py ${frame_dir} ${annotation_file}
```

to pretrain the model:

```
python cross_modal_pretrain.py ${frame_dir} ${annotation_file}
```
