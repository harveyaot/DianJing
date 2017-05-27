# DianJing
点睛 - 头条号文章标题生成工具

1. 功能：
    自动为头条的文章生成一个题目候选列表(Automatically Generate Article Title in TouTiao Style)
2. 展现形式：
    初期是linux 的客户端，后期开发一个前端页面，或者一个chrome 插件的形式存在。
3. 主要技术：
    使用encoder-decoder的技术对 头条的 摘要 和 文章对(abstract-title pair)标题进行训练
4. 数据来源：
    主要使用 头条的 数据接口，抓万级别的训练样本。

## Crawl data
1. 使用 python crawl.py 来爬取头条数据，但是需要指定头条feed 流中的 as 和 cp 两个参数，这两个参数，最好每三天更新一次，获取方法如下
   从chrom 浏览器的 network 中可以看到最新feed 流地址的这两个参数
   ![](./image/ascp.png)

## 实验日志
1. 2017/05/27  使用大约30K的训练样本，摘要-标题对，对每个汉字做100 维 embeding 使用CNN做encoder，GRU unit 的RNN 做decoer. 一天500个epoch 之后训练效果如下 ：
    * ![](./image/train_res_20170527.png)
    * 分析：
        * 可以基本的分析出描述中的关键语义
        * 但是语言可读性较差
    * 改进方向
        * 训练样本可能不足
        * 基于中文分词做，不是汉字粒度
        * LSTM 在生成长文本上的能力并不好，可以考虑基于大量语料库的language model 
