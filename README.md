# 论文笔记集

:fountain_pen: 彭正皓

<br>

> 雄关漫道真如铁，Idle boast the strong pass is a wall of iron,
> 
> 而今迈步从头越。With firm strides we are crossing its summit. 
> 
> 从头越，We are crossing its summit, 
> 
> 苍山如海，The rolling hills sea-blue, 
> 
> 残阳如血。The dying sun blood-red.

<br>

## 目录

[TOC]

规则：

* 论文从第四档标题开始，预留第二档、第三档标题分类用。
* 论文模板在本文档的隐藏内容中。

<!-- 

## 笔记模板

#### 论文标题

##### 前言

##### 方法

##### 实验设计和结果

##### 总结
-->





## 进化算法



#### Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents

本文把Novelty Search和Quality Diversity两种算法和ES结合在一起了。ES的具体原理就是施加噪声，然后利用Reward做加权得到梯度，然后更新。非常简单的东西。作者这里人为设定了Behaviour Characterization，然后用population中不同agent的BC的距离来表示Novelty，从而引入NS和QD的算法。

作者指出QD和NS可以理解成利用一群Population来进行探索，而非一个Agent单打独斗，于是这样有扩展Exploration的作用，即从专才变成了通才。



#### Learning Behavior Characterizations for Novelty Search

Behavior Characterization (BC) 将一个agent map到“行为”上，即一个表示它是什么或做了什么的向量。（这不就是我们要找的吗）本文学到的BC将用来做进化，在解迷宫问题上得到了测试。task：一个fitmess函数和一个环境。domain：一组task，用同一个BC都可以阐述。

作者指出，最简单形式的BC是：Stochastic Policy Induction for Relating Inter-task Trajectories（SPIRIT），具体而言就是：以agent在特定状态下执行特定动作的概率。设可能的state：{s1, ..., sn}和可能的action：{a1, …, am}，则BC为长度为nm的向量，

<img src="figs/gFg1D5tubEelXtvbGk3MsMgqQsOH6ERpj1WGVyskwL1kDaZosx4a8nacFwyo4oBF5hhV5bJghR2s3MWYp9TtfXXeX-A7XABkB1B3MiONJya_eszmyFpePKVsnULUugXcCzNbhTzV.png" alt="img" style="width:50%;" /><img src="figs/LGHx6yxs9rNmtvz5wRW9zi-NWCdaoPadqrOJhhE3GonrED-BMiv39lxHvQpq_G3Jsr_vghgyMCwMzvyU-wR_VNFCrIR9pNJ-oygQlKw4k_ezbw8DRQKdZtkJJEW1FFLgBGsfETz9.png" alt="img" style="width:45%;" />

上图：一批随机Policy I，和一批随机policy S。看看怎么样才能最快的从I到S）

这有点类似于QD某篇文章说的用Agent朝向东南西北所占的时间比例作为表征。但是很不幸这个需要离散的状态和动作空间。

作者随后说，所谓Learning就是给BC施加一个权重向量。那么不同的方法就是不同的权重向量的获得方式。作者拿出一批随机的Policy和一批训练完毕的成功policy。然后看看这两批policy的默认BC。看看这两批Policy的两批BC在哪个维度上区别最大，然后对应的施加权重，那么就可以得到一个新BC。

在AURORA文（见下）中有提到这篇文章，那个作者指出本方法需要定义“成功个体”的概念，这就引入了先验知识。



#### Hierarchical Behavioral Repertoires with Unsupervised Descriptors

Antonie Cully, Yiannis Demiris

用了一个有层级的行为表征。主要任务是机械手学会画数字。在机器人领域用一批diverse的行为可以拓展机器人的适应性和鲁棒性。

<img src="figs/CAbJcEc1haFHf4j4P5ecuSy_Ec957_NdXAcIDbcTw-jutFGhy3vjdLbZ5JI6Lv09V7IokwifIJtIjFUG2Rjv38CA3bf4eWGU3nkP77VXkHbg9MskhO49FJsdHEanm4FV-hS1fes0.png" alt="img" style="zoom: 25%;" />

<img src="figs/EYOIJ2j4JsvUw_ocCzvGh8Xr-9HjIrhjDc3eKvGO_L952KCO38YU6r5yBCktHtQOhQFbL-X1O19Wt2MibyUWhwAYZdKcIKggqX7cfAxUpFxW93GVaJECfZs_2PFi7qqtEMgcIVVX.png" alt="img" style="zoom:33%;" />

假设下面那个空间表示“agent画点”，那么中间这个空间就表示下面空间的一系列行为的序列，比如从一个点到另一个点。于是中间空间就表示“agent画线”。上面这个空间就是“agent画弧”。

<img src="figs/PxHbqpFYp0hEQPDS3RCpy8JAVqoD5g1IDYIbzV5OIc_5yTGi057WL-RxI9YSBFoc5_TIn5GlewwcELNohyhXsB05TvJTfIsGZOwAaC6LEDn4AeSJClsL9wvUHYwrN4evuMEp5gkJ.png" alt="img" style="zoom:33%;" />
机械臂画数字，这个数字被转换成图像，扔进MNIST的一个autoencoder中，于是得到了这个“机械臂数字->AE内部表征”。

<img src="figs/nW4X_PJyNw34z1D_OUSkXpRORdZFJFxtyCiO3enP18Cw8sTJyKqgyAxuI0ZZ0FsL-B7qisZ09OiB5k2BEJhXKivbrulx5-k0RZ2i6Fvtm9TnnOYOQT4WSAMujOSusHL3uJ_fEApZ.png" alt="img" style="zoom:33%;" />

因为有了这种层级表征，所以当我们替换最底层的Behavioral Repertoies为其他机器人的的时候，就可以transfer了。

补充：最底层的这个空间中的点到底是什么意思？答：指的是机械手最基本的一些行为，比如说，“移动到空间(x, y)”，既代表着“一个行为”，也代表着“一个执行这个行为的机械操作”，后者我们就不管他了。

思考：这个东西主要还是在机器人层面用…因为他有一个很明确的“basic behavioral descriptor”。



#### Using Centroidal Voronoi Tessellations to Scale Up the Multidimensional Archive of Phenotypic Elites Algorithm

有点意思，摘抄一下AURORA对他的评价：

<img src="figs/v6BmvOOjGKlj3V-hwcYPOPE8MOsCgi3_Oj0WK8OJlJt75Vj80nCcjcPcDcmVy6jxNlVJHf1p7PNXnjjISDYhy0TfRTGUua3jmcMVsj5J6qkvI9jwgb2rUzKnmqdgKlcmOHq-ZGMN.png" alt="img" style="zoom: 25%;" />

AURORA的作者谈到这篇文章的时候，说为什么它的结果比直接用MAP-elite好呢？因为，假设你采样了50step，每个step有一个（x，y）的坐标，那么这个BC就是100维的。在MAP-elite中，把这个100维空间给离散化，那么必然的有大量的cell是不会被填上的（因为不符合物理原理）。而CVT-MAP-Elites这个方法只采样可能的轨迹，因此就好了很多

EA算法以前是作为优化算法用的。但是越来越多的人把它作为“diversifier”，比如扔进Novelty Search算法中去。

<img src="figs/PR6I_EZKp2_PNliIKEY8FV6i8_k7-C1E2cbeNFZe4mlpMq_4J5cKHw57pIIty3b1b0_NMTI_EwHjM2Ug7SwAgsl_AjIHDTmOFElmrpjJSX2Md3kzv_pLGATZnuWnSzZTdoc-IsZp.png" alt="img" style="zoom:33%;" />

（全局优化算法就要一个结果。多目标算法可能给出几个。population-based算法直接给你一群。）

笔记正文开始：MAP-Elits对高纬度的BC空间无能为力。本文提出的方法可以给出几个指定数目的region。因为高维度的空间可以被分成若干个区域。本文运用Centroidal Voronoi Tessellation（CVT）可以将高维空间划分成若干个区域。然后就可以将任意一个高维点放进最近的区域中去。那么，相对于直接食用MAP-Elites算法把BC空间划分成一堆网格，本方法就将这个空间划分成K个区域。（格子从网格变成了不规则形状的区域）。本文就看到这里。



#### Autonomous skill discovery with Quality-Diversity and Unsupervised Descriptors

(AURORA)

<img src="figs/XfG5KsGKLPnM5UerrnqeZ6S-SfIJzKtHlb78sKh6lyNRYjinhWHP5jeTQX-DhiEwzFAC0ySXIfmhApX0G7guF7s-6Lb_r6hZeoj7-0hi5Inw8bBP4kODEuhjmM7B8Abt41AItMM6.png" alt="img" style="zoom: 33%;" />

这个图是在ICML2019的讲座上出现的引用这篇文章的图。

简单概述了一下QD的流程：

1. 随机生成一定数量的初始agents
2. 计算BD（behavioral descriptor）
3. 根据BD归类（放进Map-elites里面）
4. 从map里挑选一个agent，mutate然后回到步骤2.

本文希望能够让robot自己了解到自己的能力，而不需要任何人为给的先验知识。本文不需要提前获取什么数据集，本文只要agent自己和环境进行互动即可。

作者指出，为什么CVT-MAP-Elite是不行的？因为，他假设你可以采样到所有可能的states。但是问题是你采样到的state跟你本身的行为有关，如果你只会做A，那么你采样到的state只有A的，而可能你“冥冥之中”是会B的，但你采样不到B的轨迹。

AURORA的初始化步骤：

1. 首先用随机agent采样得到一个dataset

2. 用DR（降维）算法训练

3. 然后把dataset拿来降维，用结果作为BD


然后开始Quality Diverse步骤：

1. 随机挑选agent
2. 计算BD（这里用学习到的DR算法作为BD）
3. 扔到MAP-Elites格子里面去

在适当的轮数之后，重新学习DR算法。补充：扔进DR算法的东西是一个100维度的向量。表示50step，每个step有个(x,y)坐标。



#### Unsupervised Learning of Goal Spaces for Intrinsically Motivated Goal Exploration

本文提到了一个KL Coverage的指标，可以用来衡量两个repertory的差异。

Intrinsic Reward用来寻找Diverse的动作。那么其实这个外界的刺激，本身可以理解成agent的任务发生了变化，即goal发生了变化，从原来的reward对应的那个task，变成了task+novel。因此，intrinsic reward可以理解成在goal空间中给了一个新的goal。

这里说如何才能自动生成goal呢？其实这里的goal和我们说的BC可以等价。因为这里的goal指的是agent以某种方式运动，或者说环境发生了某种形式的改变。那其实，goal空间不就是我们说的behavior空间嘛。总之就是要有一个空间来表示一个agent，或者一类行为，或者一类环境发生的改变。（agent X 环境 = 行为）

作者指出：forms of random goal exploration are a form of intrinsically motivated exploration

那么怎么选取不同的goal呢？这就变成了一个meta-learning的问题。除了我们随机的选择goal，就像随机的鼓励intrinsic reward一样，其实还有别的更好的方法（编者：咦？这不就是我想要的有目标的去调整行为嘛！看来meta-learning也要看一看。）

众多goal的选取的方法都要求用户给定一个goal的形式，并且给一个函数来衡量agent是否已经达成了goal。这就有问题了：

1. 能不能让agent用无监督学习的方法，在它实现了这个goal之前，就已经学习到了这个goal的一种表征？
2. 对于一个学习到的goal的表征，如何选取“interesting goal”呢？这个goal不能太夸张，太不合实际。
3. 这个无监督学习的方法效率不能太差。

本文提出IMGEP-UGL算法，分成了深度表示学习和goal探索两个步骤。第一步先学会这个世界的“configure”，就像婴儿睁眼看世界一样，第二步从这个世界模型中采样出goal来进行下一步活动（婴儿学会爬行和走路）。

Intrinsically Motivated Goal Exploration Processes (IMGEPs) 一种启发式算法，从高维连续动作空间中学习forward和inverse的控制模型，以解决robotic问题。

很有意思的点，作者将Exploration Process理解成：

1. Context：一些agent不能操控的东西，比如环境的dynamic
2. Parameterization：agent能够改变的东西，比如policy的参数
3. Outcome：对行为的一种characterization
4. phenomenon dynamics：从parameter到outcome的映射关系。

在developmental robotics中，希望学到一个forward模型：context+param->outcome和一个inverse模型：context+outcome->param。一个简单的想法就是用各种随机的param来拟合这两个模型。但是在参数空间中有大量的参数组合是没用的。所以IMGEP就是用来解决这个问题：它希望找到一种sample参数的方法，以得到最有价值（informative）的那些sample。

IMGEP定义了：

1. Goal Space：基本等价于Outcome，指我们希望agent做什么样的事情
2. Goal Policy：关于goal的概率分布，采样用。（这个policy不是那个agent的policy）
3. A set of Goal-parameterized Cost Functions：给定一个outcome，给我一个fitness/performance/reward/cost。
4. Meta-Policy：给定一个goal和context，给我一个policy（即给我一组param来得到一个policy）。具体而言是：$f(goal, context) =argmin_\theta C_{goal}(Outcome(\theta, context))$

编者注：这套流程就是我要的retrival啊！但是现在暂时还不说这个，我想知道的事outcome的表征。

作者指出，outcome space的结构是人为给定的，这样不好，所以他打算用深度表示学习的方法来自动学习这个表征。也就是AE、VAE等方法。

它的网络：

<img src="figs/WB7tqoz2CYrOxvNpZuSgnddvxxq7tgWX6tvPAVVfnoGh3VkdmHtGE72s2cb1_ZnB1SOjR7TZoqRpNbcXNkS88VdSl-9PhKprC8fK8HH6AaSW4OTwSvYwFtHbcgl-geYCSgB1v0Ni.png" alt="img" style="zoom:25%;" />

输入的是随机抽取的一些图像。我的问题是，输入的是一个step的东西，为什么中间的这个表征可以代表一个agent？答：因为“到达某个state”本身就是“goal” 啊。哦，所以看来看去这个文章只是在探究一个机械臂能够伸到哪些地方。这并不是我想要的对行为的characterization。



#### Policy Distillation

各种策略如何融合。需要训练。这个可能对我们最后retrive那个步骤有用。



#### Unsupervised Learning and Exploration of Reachable Outcome Space

唯一一篇引用AURORA的文章。应该说已经站在了最前沿。

<img src="figs/RuvQhwSuMYUM21-3MiVgvTM26LCPp8ki5pfjoRc4FmhL_yHSJThlkloXHQSSDcSqQ9kJLdbfbqxh1JfG4PC4j0HC_7KpwF1k2RMbChFctx5PPrkg6xtTns9TLdSmgFEF5drMGHCM.png" alt="img" style="zoom:25%;" />

作者指出AURORA的observation和系统的low-dim state是直接相关的（比如robot在移动自己），然而如果robot的state和env的state的低维形式并不能直接拿到的话就不行了。

本文使用novelty和suprise两个概念。novelty：outcome space的距离。surprise：AE的重构误差。

本文引用了Novelty Search作为formulation。简单的概述NS如下：

1. 定义一个从policy到BD（behaviour description）的函数
2. 对于一个policy，计算它和K临近policy的distance（对应BD的欧几里得距离）
3. 留下最novel的agent。mutate，回到步骤2。

作者用了最后一帧的obs来做AE的输入。这点我非常的不认同。

然后作者把novelty和suprise两个东西结合起来了，这点非常好。因为引入了AE才有的suprise，把它加入进来很合理（也很新颖）。

作者在结论部分说，这个AE可以用来做retrieveal，因为只要把你想要的最后一帧outcome输入进去，得到内部表征，然后在population中寻找最接近它的那个agent，就完事儿了。



#### Behavioral diversity with multiple behavioral distances

（2013）混合了多种Bahaviour Descriptor

Behavioral Similarity Measure（BSM）是进化机器人学很重要的问题。作者说其实在用它做进化算法的时候不一定要选择一个，我全都要。

在进化机器人学（Evolutionary Robotics，ER）中，观察到可能很多的genotypes会带来相似的行为，而只有一点点差别的genotypes会带来很不同的行为。所以只考虑genotypes的多样性是不够的，真正重要的是在行为空间中的多样性。有各种各样的BSM来衡量这个多样性，但是我们很难找到最优的那个。所以为何不把它们融合起来？

第一种方法就是求平均值。任何一个BSM，将其除以最大值之后，求平均值。

第二种方法就是选定一个BSM，然后在一定的时间之后随机的跳到另一个BSM。

很无聊啊这文章…………但我们可以看看他有什么五种BSM，有没有值得学习的：

1. adhoc：最后一刻的位置。
2. hamming：最后4000时刻，存储每个传感器的值，这样你就有4000*x个值了，这作为一个behavior的表征。然后对两个行为之间求hamming距离（bit的差异数目）。
3. trajectory：50时刻的robot的位置
4. entropy：比较复杂，而且看起来挺厉害的，看看实验结果如何，好的话可以拿过来。一般，不如trajectory。



#### Evolving a Behavioral Repertoire for a Walking Robot

**（一把学全部！！）**

一组简单的控制器，每个负责往一个方向走，那么把他们融合起来就会得到一个掌握了往各个方向走的技能的agent。

因此，本文的核心任务就是“evolving a repertoire”。学会各种简单的任务，比学会一个复杂的任务要简单。独立的学习不同任务可能会很昂贵。因此作者提出的Transferability-based Behavioral Repertoire Evolution algorithm (TBR-Evolution)将问题从一个agent的许多许多技能变成了学习许多许多个各不相同的agent。

(TODO)

