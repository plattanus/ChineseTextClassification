# ChineseTextClassification

中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch。

##### 数据来源

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)

##### 环境

**python 3.7 、 pytorch 、  tqdm 、 sklearn、  tensorboardX**

##### 数据集

**从**[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

**类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。**

**数据集划分：**

| **数据集** | **数据量** |
| ---------------- | ---------------- |
| **训练集** | **18万**   |
| **验证集** | **1万**    |
| **测试集** | **1万**    |

* 如果用字，按照我数据集的格式来格式化你的数据。
* **如果用词，提前分好词，词之间用空格隔开，**`python run.py --model TextCNN --word True`
* **使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。**

##### 效果

| **模型**        | **acc**    | **备注**                              |
| --------------------- | ---------------- | ------------------------------------------- |
| **TextCNN**     | **91.03%** | **Kim 2014 经典的CNN文本分类**        |
| **DPCNN**       | **90.88%** | **BiLSTM**                            |
| **TextRNN**     | **90.41%** | **BiLSTM+Attention**                  |
| **TextRNN_Att** | **90.48%** | **BiLSTM+池化**                       |
| **TextRCNN**    | **91.01%** | **bow+bigram+trigram， 效果出奇的好** |
| **FastText**    | **91.58%** | **深层金字塔CNN**                     |

#### RUN

```
# 训练并测试：
# TextCNN
python run.py --model TextCNN

# TextRNN
python run.py --model TextRNN

# TextRNN_Att
python run.py --model TextRNN_Att

# TextRCNN
python run.py --model TextRCNN

# FastText
python run.py --model FastText --embedding random 

# DPCNN
python run.py --model DPCNN

```

### 对应论文

**[1] Convolutional Neural Networks for Sentence Classification**
**[2] Recurrent Neural Network for Text Classification with Multi-Task Learning**
**[3] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification**
**[4] Recurrent Convolutional Neural Networks for Text Classification**
**[5] Bag of Tricks for Efficient Text Classification**
[6] Deep Pyramid Convolutional Neural Networks for Text Categorization
