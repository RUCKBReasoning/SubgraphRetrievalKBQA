How to generate weakly supervised data:  
Corresponding to  5.1 Weakly Supervised Pre-Training

1. run ```search_to_get_path.py``` to enumerate all path from topic entity to answer   
请参考generate_paths的函数，其将搜索长度不超过最短路+1的由entity构成的路径，pair_max 限制了一对头实体和答案的路径个数、
path_max 限制了头实体和答案集合整体的路径个数，两个参数都是为了避免搜索空间过大设置的，理论上无限制。

2. run ```path_to_relation_list.py``` to convert path to relation list, instantiate it, and caculate its score.  
将抽出的路径转化成对应的 relation 构成的 list, 即relation path, 将其从头实体出发实例化，查看其叶子结点对应答案集合的Hit, 从中挑出得分较高的作为正例。
