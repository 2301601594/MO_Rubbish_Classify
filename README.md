# MO：手写数字识别与垃圾分类——垃圾分类
## 程序结构
```bash
.
├── README.md
├── checkpoints
├── convert.py
├── dataset.py
├── finetune
├── garbage_26x100
├── logs
├── main.py
├── my_models
├── predict.py
├── pretrain_weights
└── utils.py
```
- checkpoints：保存训练好的权重
- convert.py：将pth文件转换为pt文件
- dataset.py：定义数据集结构
- finetune：针对对应模型进行微调
- garbage_26x100：数据集
- logs：tensorboard文件，记录模型训练指标
- main.py：训练主程序
- my_models：定义使用的模型
- predict.py：使用模型进行预测
- pretrain_weights：部分模型的预训练权重
- utils.py：部分辅助函数

## 附加题——移植到安卓
参考文章：https://blog.csdn.net/weixin_46573159/article/details/138377842

程序代码：https://github.com/2301601594/MobileNetV3-Android