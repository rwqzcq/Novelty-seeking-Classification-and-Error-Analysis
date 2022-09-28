# SentimentClassificationErrorAnalyse
Sentiment Classification Error Analyse

# 安装说明

```shell
pip install -r requirements.txt
pip install en_core_web_sm-3.4.0-py3-none-any.whl
```
# 运行说明

```shell
nohup python -u 2_train.py > gru.log &
```

# 误差分析

偏差(bias)与方差(variance)

1. 把错误的挑出来做统计分析，找到出错的样本有哪一些特征，然后把这些特征融入到之前的特征里面去

TODO 比较DEV与TEST中的accuracy，参考[这里](https://www.pianshen.com/article/2957376531/)

## 混淆矩阵

```text
[tester] 
ClassifyFPreRecMetric: f=0.9525, pre=0.9525, rec=0.9525
ConfusionMatrixMetric: confusion_matrix=
target    0       1     all
  pred
     0  241      13     254
     1    6     140     146
   all  247     153     400
```

# 参考链接

1. 分类问题（六）误差分析. https://www.cnblogs.com/zackstang/p/12332109.html
2. 使用scikit-learn中的metrics.plot_confusion_matrix混淆矩阵函数分析分类器的误差来源. https://blog.csdn.net/cxx654/article/details/107296343/
3. AI学习笔记——机器学习中误差分析的几个关键概念. https://www.pianshen.com/article/2957376531/
4. 吴恩达机器学习图解笔记P50. https://www.bilibili.com/video/BV1XW411u7te?p=50&vd_source=526a7830d39d06de69cc3bb950efbfcd