# Stock_prediction
深度学习-贵州茅台股票预测     PCA、FA、RNN、LSTM、GRU

2021.01参加光华案例大赛的代码，对贵州茅台股价的影响因素进行建模，并使用时间序列进行预测。
> 真实数据和预测数据对比结果如下：
 
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784854-e6015442-cee2-4a68-9390-3d34fb8ba292.png"/></div>
  
> 指标概述
采用多级指标对影响股价的因素进行分析，三个一级指标分别为公司模块、股市模块、总体经济模块，一级指标对应的二级指标如下所示：

* 股市模块

.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130783781-67d4711a-adc4-4a12-a079-cf3a3b47e3c2.png"/></div>
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130783795-dba56b66-bfba-4c9b-a0fe-362720b272d4.png"/></div>

 
* 公司模块
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784021-2fbe2a28-a735-4259-b515-2554dce7533f.png"/></div>
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784014-315682b4-f82a-4745-a1e3-bca819112963.png"/></div>
 
* 总体经济模块
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784045-1bbe5a66-b4d1-4766-b024-b34c60c47820.png"/></div>
 
> 因子分析

采用FA对上述66个二级指标分析如下图所示，可以看出各个指标之间存在一定的相关性，因子分析表明存在至少5个代表性的因子，可将不同的二级指标按照相关性进行划分。
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784075-d2b4176b-a068-4c21-82c5-30916d5981e2.png"/></div>

> 模型数据
对2016.06.29~2021.01.21的工作日的上述指标数据进行统计，数据来源于wind以及中国统计年鉴。

> 模型搭建
* 标准化
采用Max-Min标准化，由于不同指标的量纲和量纲单位差异较大，会影响到数据分析的结果，为了消除指标之间的量纲影响，需要进行数据标准化处理，以解决数据指标之间的可比性。原始数据经过数据标准化处理后，各指标处于同一数量级，适合进行综合对比评价。对数据标准化处理，便于消除特征之间的差异性。
Max-Min标准化是对原始数据进行线性变换，将值映射到[0，1]之间。对某二级指标下的所有数据，第i 个数据的变换方式为：
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784760-38ae6ba1-b408-42b6-b446-de5345ea5656.png"/></div>
标准化得到的第i个数据为yi，yi∈[0, 1]且无量纲。

* 去中心化
采用去中心化，对数据进行中心化预处理，可以增加基向量的正交性。通过中心化和标准化处理，得到的数据服从均值为0，标准差为1的正态分布。
数据去中心化的方式为变量减去均值，使的去中心化后的数据均值为0.

* PCA降维
为了降低数据维度，提高运算速度，并且尽可能减少数据的损失，使用主成分分析法对输入数据进行降维，调用sklearn中的PCA模块对输入数据进行降维，采用常用的95%的累计贡献率，可将66维输入指标降至5维。
PCA 算法使得样本的采样密度增大，缓解了维度灾难。当数据受到噪声影响时，最小特征值对应的特征向量往往与噪声有关，将它们舍弃能在一定程度上起到降噪的效果，另外，降维之后的数据可以使得各特征相互独立。

* RNN模型
RNN是可以用于处理时间序列数据的神经网络，所谓时间序列数据，是指在不同时间点上收集到的数据，这类数据反映了某一事物或现象随时间的变化状态或程度。股票数据是一种时间序列数据，即某天的股价可能与之前n天的相关数据有关。
RNN结构图如下所示：
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784788-6f0747db-1aba-4e6c-bf46-fbddac7abed2.png"/></div>


假设第x+1天的股价与前n天的指标数据相关，以第x-n天 ~ 第 x天指标数据作为输入，第x+1天股价作为输出，构建RNN模型。
n的取值范围为5，10，15，20，25，30，40，50，计算不同n下的均方误差，结果如下：
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784824-7201c16b-87cf-4367-941f-ad64c14b3f51.png"/></div>

计算不同n下的均方根误差，结果如下：
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784837-bffeb670-38a0-4cd1-9fc7-88d7b8814ee9.png"/></div>

计算不同n下的平均绝对误差，结果如下：
.<div align=center><img src="https://user-images.githubusercontent.com/58354216/130784846-e02a0ddd-fa3c-43cb-abe3-e7bc1cc9b3f1.png"/></div>

由于模型复杂度随输入数据规模增加而增加，因此，选取n=30为较优解，建立贵州茅台股价预测循环神经网络，以x-30~x天的预处理后的指标数据作为输入，第x+1天的股价作为输出，神经网络采用两层RNN及一层全连接层。

> 模型改进
  
* 采用LSTM网络改进RNN模型，避免长期依赖问题，进而可以采用GRU模型提升计算效率。
* 增加社会模块一级指标，包含的二级指标可以包含公司舆情、监管、评级等，可以采用大数据分析与数据挖掘相关技术挖掘舆情信息。





