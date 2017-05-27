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
1. 使用 python crawl.py 来爬取头条数据，但是需要
