# **Awesome ICML 2024 Graph Paper Collection**

This repo contains a comprehensive compilation of **graph and/or GNN** papers that were accepted at the [International Conference on Machine Learning 2024](https://openreview.net/group?id=ICML.cc/2024/Conference). Graph or Geometric machine learning possesses an indispensable role within the domain of machine learning research, providing invaluable insights, methodologies, and solutions to a diverse array of challenges and problems. 

**Short Overview**: We've got around 250 papers focusing on Graphs and GNNs in ICML'24.
The core themes of this year include [equivariant GNNs](#Equivariant), [OODs](#OOD), [diffusions](#Diffusion), [heterophily](#Heterophily), [expressivity](#Expressivity), and [clustering](#Clustering). There's also a good amount of [casual graph](#casual) works, more than I expected. We have some very good [physics-inspired](#PDE) research too. 
A lot of [application](#apps) papers are available, although I expected to see more in [molecular, chemical GNNs](#molecular) and [GFlowNets](#GFlowNets). [Reinforcement learning](#RL) also had a good boost this year.
**Have a look and throw me a review (and, a star ⭐, maybe!)** Thanks!


## **All Topics:** 

<details>
  <summary><b>View Topic list!</b></summary>

- [GNN Theories](#theories)
  - [Weisfeiler Leman](#Weisfeiler-Leman )
  - [Heterophily](#Heterophily)
  - [Hypergraph](#Hypergraph)
  - [Expressivity](#Expressivity)
  - [Generalization](#Generalization)
  - [Equivariant Graph Neural Networks](#Equivariant)
  - [Out-of-Distribution](#OOD)
  - [Diffusion](#Diffusion)
  - [Graph Matching](#GraphMatching)
  - [Contrastive Learning](#ContrastiveLearning)
  - [Clustering](#Clustering)
  - [Foundational Models](#FM)
  - [Message Passing Neural Networks](#MPNN)
  - [Graph Transformers](#GraphTransformers)
  - [Class Imbalance](#ClassImbalance)
  - [Optimal Transport](#OptimalTransport)
  - [Graph Generation](#ggen)
  - [Unsupervised Learning](#UL)
  - [Meta-learning](#GraphMeta-learning)
  - [Disentanglement](#Disentanglement)
  - [Others](#GNNT-others)
- [GNNs for PDE/ODE/Physics](#PDE)
- [Graph and Large Language Models/Agents](#LLM)
- [Knowledge Graph and Knowledge Graph Embeddings](#KG)
- [GNN Applications](#apps)
- [Spatial and/or Temporal GNNs](#SpatialTemporalGNNs)
- [Explainable AI](#xai)
- [Reinforcement Learning](#rl)
- [Graphs, Molecules and Biology](#molecular)
- [GFlowNets](#GFlowNets)
- [Casual Discovery and Graphs](#casual)
- [Federated Learning, Privacy, Decentralization](#FL)
- [Scene Graphs](#SceneGraphs)
- [Position Papers](#PositionPapers)
- [Others](#Others)

</details>



<a name="theories" />

## GNN Theories 

<a name="Weisfeiler-Leman " />

#### Weisfeiler Leman 
- [Weisfeiler-Leman at the margin: When more expressivity matters](https://openreview.net/forum?id=HTNgNt8CTJ)
- [Aligning Transformers with Weisfeiler-Leman](https://openreview.net/forum?id=4FJJfYjUQR)
- [Weisfeiler Leman for Euclidean Equivariant Machine Learning](https://openreview.net/forum?id=ApRKrKZJSk)


#### Heterophily
- [Understanding Heterophily for Graph Neural Networks](https://openreview.net/forum?id=wK9RvVmi7u)
- [How Universal Polynomial Bases Enhance Spectral Graph Neural Networks: Heterophily, Over-smoothing, and Over-squashing](https://openreview.net/forum?id=Z2LH6Va7L2)
- [Unraveling the Impact of Heterophilic Structures on Graph Positive-Unlabeled Learning](https://openreview.net/forum?id=NCT3w7VKjo)
- [Multi-Track Message Passing: Tackling Oversmoothing  and Oversquashing in Graph Learning via Preventing Heterophily Mixing](https://openreview.net/forum?id=1sRuv4cnuZ)
- [Sign is Not a Remedy: Multiset-to-Multiset Message Passing for Learning on Heterophilic Graphs](https://openreview.net/forum?id=dGDFZM018a)
- [Mitigating Oversmoothing Through Reverse Process of GNNs  for Heterophilic Graphs](https://openreview.net/forum?id=RLA4JTckXe)


#### Hypergraph
- [Fast Algorithms for Hypergraph PageRank with Applications to Semi-Supervised Learning](https://openreview.net/forum?id=sfQH4JJ4We)
- [Hypergraph-enhanced Dual Semi-supervised Graph Classification](https://openreview.net/forum?id=M5ne8enLcr)
- [Relational Learning in Pre-Trained Models: A Theory from Hypergraph Recovery Perspective](https://openreview.net/forum?id=puSMYmHmJW)

#### Expressivity
- [Expressivity and Generalization: Fragment-Biases for Molecular GNNs](https://openreview.net/forum?id=rPm5cKb1VB)
- [On the Expressive Power of Spectral Invariant Graph Neural Networks](https://openreview.net/forum?id=kmugaw9Kfq)
- [The Expressive Power of Path-Based Graph Neural Networks](https://openreview.net/forum?id=io1XSRtcO8)
- [On the Expressive Power of Spectral Invariant Graph Neural Networks](https://openreview.net/forum?id=kmugaw9Kfq)
- [An Empirical Study of Realized GNN Expressiveness](https://openreview.net/forum?id=WIaZFk02fI)

#### Generalization
- [Weisfeiler-Leman at the margin: When more expressivity matters](https://openreview.net/forum?id=HTNgNt8CTJ)
- [Generalization Error of Graph Neural Networks in the Mean-field Regime](https://openreview.net/forum?id=8h0x12p3zq)
- [On the Generalization of Equivariant Graph Neural Networks](https://openreview.net/forum?id=Yqj3DzIC79)
- [Enhancing Size Generalization in Graph Neural Networks through Disentangled Representation Learning](https://openreview.net/forum?id=0NdU4y9dWC)
- [What Improves the Generalization of Graph Transformers? A Theoretical Dive into the Self-attention and Positional Encoding](https://openreview.net/forum?id=mJhXlsZzzE)
- [Semantically-correlated memories in a dense associative model](https://openreview.net/forum?id=l0OGoZPZuC)
- [Improved Stability and Generalization Guarantees of the Decentralized SGD Algorithm](https://openreview.net/forum?id=JKPhWzp7Oi)
- [PAC-Bayesian Generalization Bounds for Knowledge Graph Representation Learning](https://openreview.net/forum?id=sOyJSNUrzQ)
- [Beyond the Federation: Topology-aware Federated Learning for Generalization to Unseen Clients](https://openreview.net/forum?id=2zLt2Odckx)
- [A Circuit Domain Generalization Framework for Efficient Logic Synthesis in Chip Design](https://openreview.net/forum?id=1KemC8DNa0)


<a name="Equivariant" />

#### Equivariant Graph Neural Networks
- [On the Generalization of Equivariant Graph Neural Networks](https://openreview.net/forum?id=Yqj3DzIC79)
- [Improving Equivariant Graph Neural Networks on Large Geometric Graphs  via Virtual Nodes Learning](https://openreview.net/forum?id=wWdkNkUY8k)
- [Equivariant Graph Neural Operator for Modeling 3D Dynamics](https://openreview.net/forum?id=dccRCYmL5x)
- [Generalist Equivariant Transformer Towards 3D Molecular Interaction Learning](https://openreview.net/forum?id=dWxb80a0TW)
- [Topological Neural Networks go Persistent, Equivariant, and Continuous](https://openreview.net/forum?id=ELFZWG9C7l)
- [Interpreting Equivariant Representations](https://openreview.net/forum?id=vFk9fqXLst)
- [Graph Automorphism Group Equivariant Neural Networks](https://openreview.net/forum?id=vjkq5fwsj3)
- [Weisfeiler Leman for Euclidean Equivariant Machine Learning](https://openreview.net/forum?id=ApRKrKZJSk)
- [Subequivariant Reinforcement Learning in 3D Multi-Entity Physical Environments](https://openreview.net/forum?id=hQpUhySEJi)



<a name="OOD" />

#### Out-of-Distribution
- [Graph Structure Extrapolation for Out-of-Distribution Generalization](https://openreview.net/forum?id=Xgrey8uQhr)
- [When and How Does In-Distribution Label Help Out-of-Distribution Detection?](https://openreview.net/forum?id=knhbhDLdry)
- [Graph Out-of-Distribution Detection Goes Neighborhood Shaping](https://openreview.net/forum?id=pmcusTywXO)
- [Context-Guided Diffusion for Out-of-Distribution Molecular and Protein Design](https://openreview.net/forum?id=8NfHmzo0Op)
- [Disentangled Graph Self-supervised Learning for Out-of-Distribution Generalization](https://openreview.net/forum?id=OS0szhkPmF)
- [Bounded and Uniform Energy-based Out-of-distribution Detection for Graphs](https://openreview.net/forum?id=mjh7AOWozN)
- [Prometheus: Out-of-distribution Fluid Dynamics Modeling with Disentangled Graph ODE](https://openreview.net/forum?id=JsPvL6ExK8)

<a name="GraphMatching" />

#### Graph Matching
- [Robust Graph Matching when Nodes are Corrupt](https://openreview.net/forum?id=WJn1BAx9aj)
- [Effective Federated Graph Matching](https://openreview.net/forum?id=rSfzchjIYu)


<a name="ContrastiveLearning" />

#### Contrastive Learning
- [S3GCL: Spectral, Swift, Spatial Graph Contrastive Learning](https://openreview.net/forum?id=znKAWRZSF9)
- [Perfect Alignment May be Poisonous to Graph Contrastive Learning](https://openreview.net/forum?id=wdezvnc9EG)
- [UniCorn: A Unified Contrastive Learning Approach for Multi-view Molecular Representation Learning](https://openreview.net/forum?id=2NfpFwJfKu)
- [Community-Invariant Graph Contrastive Learning](https://openreview.net/forum?id=dskLpg8WFb)

### Diffusion
- [Graph Adversarial Diffusion Convolution](https://openreview.net/forum?id=ICvWruTEDH)
- [Editing Partially Observable Networks via Graph Diffusion Models](https://openreview.net/forum?id=2cEhQ4vtTf)
- [Cluster-Aware Similarity Diffusion for Instance Retrieval](https://openreview.net/forum?id=qMG3OK7Xcg)
- [Neurodegenerative Brain Network Classification via Adaptive Diffusion with Temporal Regularization](https://openreview.net/forum?id=GTnn6bNE3j)
- [Hyperbolic Geometric Latent Diffusion Model for Graph Generation](https://openreview.net/forum?id=6OkvBGqW62)
- [Diffuse, Sample, Project: Plug-And-Play Controllable Graph Generation](https://openreview.net/forum?id=ia0Z8d1DbY)
- [Graph Generation with Diffusion Mixture](https://openreview.net/forum?id=cZTFxktg23)
- [Learning Iterative Reasoning through Energy Diffusion](https://openreview.net/forum?id=CduFAALvGe)


#### Clustering
- [Multi-View Clustering by Inter-cluster Connectivity Guided Reward](https://openreview.net/forum?id=uEx2bSAJu8)
- [Dynamic Spectral Clustering with Provable Approximation Guarantee](https://openreview.net/forum?id=coP4kPdhKr)
- [EDISON: Enhanced Dictionary-Induced Tensorized Incomplete Multi-View Clustering with Gaussian Error Rank Minimization](https://openreview.net/forum?id=fiugPLSXjK)
- [Pruned Pivot: Correlation Clustering Algorithm for Dynamic, Parallel, and Local Computation Models](https://openreview.net/forum?id=saP7s0ZgYE)
- [LSEnet: Lorentz Structural Entropy Neural Network for Deep Graph Clustering](https://openreview.net/forum?id=L6SRXG92s6)
- [A Near-Linear Time Approximation Algorithm for Beyond-Worst-Case Graph Clustering](https://openreview.net/forum?id=MSFxOMM0gK)
- [Cluster-Aware Similarity Diffusion for Instance Retrieval](https://openreview.net/forum?id=qMG3OK7Xcg)
- [Combinatorial Approximations for Cluster Deletion: Simpler, Faster, and Better](https://openreview.net/forum?id=FpbKoIPHxb)



<a name="MPNN" />

#### Message Passing Neural Networks
- [Verifying message-passing neural networks via topology-based bounds tightening](https://openreview.net/forum?id=nAoiUlz4Bf)
- [Sign is Not a Remedy: Multiset-to-Multiset Message Passing for Learning on Heterophilic Graphs](https://openreview.net/forum?id=dGDFZM018a)
- [PANDA: Expanded Width-Aware Message Passing Beyond Rewiring](https://openreview.net/forum?id=J1NIXxiDbu)
- [Pluvial Flood Emulation with Hydraulics-informed Message Passing](https://openreview.net/forum?id=kIHIA6Lr0B)
- [On dimensionality of feature vectors in MPNNs](https://openreview.net/forum?id=UjDp4Wkq2V)


<a name="FM" />

#### Foundational Models
- [Position: Graph Foundation Models Are Already Here](https://openreview.net/forum?id=Edz0QXKKAo)


<a name="GraphTransformers" />

#### Graph Transformers
- [What Improves the Generalization of Graph Transformers? A Theoretical Dive into the Self-attention and Positional Encoding](https://openreview.net/forum?id=mJhXlsZzzE)
- [Triplet Interaction Improves Graph Transformers: Accurate Molecular Graph Learning with Triplet Graph Transformers](https://openreview.net/forum?id=iPFuWc1TV2)
- [Aligning Transformers with Weisfeiler-Leman](https://openreview.net/forum?id=4FJJfYjUQR)
- [Less is More: on the Over-Globalizing Problem in Graph Transformers](https://openreview.net/forum?id=uKmcyyrZae)
- [Comparing Graph Transformers via Positional Encodings](https://openreview.net/forum?id=va3r3hSA6n)
- [Subgraphormer: Unifying Subgraph GNNs and Graph Transformers via Graph Products](https://openreview.net/forum?id=6djDWVTUEq)


<a name="ClassImbalance" />

#### Class Imbalance
- [Class-Imbalanced Graph Learning without Class Rebalancing](https://openreview.net/forum?id=pPnkpvBeZN)
- [Automated Loss function Search for Class-imbalanced Node Classification](https://openreview.net/forum?id=O1hmwi51pp)


<a name="OptimalTransport" />

#### Optimal Transport
- [OT-CLIP: Understanding and Generalizing CLIP via Optimal Transport](https://openreview.net/forum?id=X8uQ1TslUc)
- [Optimal Transport for Structure Learning Under Missing Data](https://openreview.net/forum?id=09Robz3Ppy)
- [Parameter Estimation in DAGs from Incomplete Data via Optimal Transport](https://openreview.net/forum?id=kXde6Qa6Uy)
- [Generalized Sobolev Transport for Probability Measures on a Graph](https://openreview.net/forum?id=0GC0NG6Orr)



<a name="ggen" />

#### Graph Generation
- [Scene Graph Generation Strategy with Co-occurrence Knowledge and Learnable Term Frequency](https://openreview.net/forum?id=tTq3qMkJ8w)
- [Hyperbolic Geometric Latent Diffusion Model for Graph Generation](https://openreview.net/forum?id=6OkvBGqW62)
- [Diffuse, Sample, Project: Plug-And-Play Controllable Graph Generation](https://openreview.net/forum?id=ia0Z8d1DbY)
- [Graph Generation with Diffusion Mixture](https://openreview.net/forum?id=cZTFxktg23)
- [On the Role of Edge Dependency in Graph Generative Models](https://openreview.net/forum?id=0XDO74NlOd)


<a name="UL" />

#### Unsupervised Learning
- [Unsupervised Episode Generation for Graph Meta-learning](https://openreview.net/forum?id=9zdTOOgutk)
- [Unsupervised Representation Learning of Brain Activity via Bridging Voxel Activity and Functional Connectivity](https://openreview.net/forum?id=nOjZfpLyh1)
- [Unsupervised Parameter-free Simplicial Representation Learning with Scattering Transforms](https://openreview.net/forum?id=wmljUnbjy6)
- [Tackling Prevalent Conditions in Unsupervised Combinatorial Optimization: Cardinality, Minimum, Covering, and More](https://openreview.net/forum?id=6n99bIxb3r)

<a name="GraphMeta-learning" />

#### Graph Meta-learning
- [Unsupervised Episode Generation for Graph Meta-learning](https://openreview.net/forum?id=9zdTOOgutk)


#### Disentanglement
- [Prometheus: Out-of-distribution Fluid Dynamics Modeling with Disentangled Graph ODE](https://openreview.net/forum?id=JsPvL6ExK8)
- [Disentangled Graph Self-supervised Learning for Out-of-Distribution Generalization](https://openreview.net/forum?id=OS0szhkPmF)
- [Enhancing Size Generalization in Graph Neural Networks through Disentangled Representation Learning](https://openreview.net/forum?id=0NdU4y9dWC)
- [Disentangled Continual Graph Neural Architecture Search with Invariant Modular Supernet](https://openreview.net/forum?id=Hg7C5YYifi)
- [Enhancing Size Generalization in Graph Neural Networks through Disentangled Representation Learning](https://openreview.net/forum?id=0NdU4y9dWC)


<a name="GNNT-others" />

#### GNN Theory : Others
- [GNNs Also Deserve Editing, and They Need It More Than Once](https://openreview.net/forum?id=rIc9adYbH2)
- [Learning Divergence Fields for Shift-Robust Graph Representations](https://openreview.net/forum?id=jPaEOH56JB)
- [Probabilistic Routing for Graph-Based Approximate Nearest Neighbor Search](https://openreview.net/forum?id=pz4B2kHVKo)
- [Efficient Contextual Bandits with Uninformed Feedback Graphs](https://openreview.net/forum?id=0vozy8vstt)
- [Efficient Contrastive Learning for Fast and Accurate Inference on Graphs](https://openreview.net/forum?id=vsy21Xodrt)
- [Graph Geometry-Preserving Autoencoders](https://openreview.net/forum?id=acTLXagzqd)
- [Stereographic Spherical Sliced Wasserstein Distances](https://openreview.net/forum?id=vLtVGtEz5h)
- [Perfect Alignment May be Poisonous to Graph Contrastive Learning](https://openreview.net/forum?id=wdezvnc9EG)
- [Sparse-IFT: Sparse Iso-FLOP Transformations for Maximizing Training Efficiency](https://openreview.net/forum?id=X8Ha2NiQcy)
- [Prospector Heads: Generalized Feature Attribution for Large Models & Data](https://openreview.net/forum?id=PjVqEErDgK)
- [Collective Certified Robustness against Graph Injection Attacks](https://openreview.net/forum?id=DhxZVq1ZOo)
- [Graph Distillation with Eigenbasis Matching](https://openreview.net/forum?id=DYN66IJCI9)
- [Graph Neural Networks Use Graphs When They Shouldn't](https://openreview.net/forum?id=fSNHK7mu3j)
- [Quantum Positional Encodings for Graph Neural Networks](https://openreview.net/forum?id=IW45Dr1Kxi)
- [SLOG: An Inductive Spectral Graph Neural Network Beyond Polynomial Filter](https://openreview.net/forum?id=0SrNCSklZx)
- [How Interpretable Are Interpretable Graph Neural Networks?](https://openreview.net/forum?id=F3G2udCF3Q)
- [Graph Neural Networks with a Distribution of Parametrized Graphs](https://openreview.net/forum?id=VyfEv6EjKR)
- [Cooperative Graph Neural Networks](https://openreview.net/forum?id=ZQcqXCuoxD)
- [Generalization Error of Graph Neural Networks in the Mean-field Regime](https://openreview.net/forum?id=8h0x12p3zq)
- [Networked Inequality: Preferential Attachment Bias in Graph Neural Network Link Prediction](https://openreview.net/forum?id=GhPFmTJNfj)
- [EvoluNet: Advancing Dynamic Non-IID Transfer Learning on Graphs](https://openreview.net/forum?id=anM1M5aoM8)
- [How Graph Neural Networks Learn: Lessons from Training Dynamics](https://openreview.net/forum?id=Dn4B53IcCW)
- [Homomorphism Counts for Graph Neural Networks: All About That Basis](https://openreview.net/forum?id=zRrzSLwNHQ)
- [HGCN2SP: Hierarchical Graph Convolutional Network for Two-Stage Stochastic Programming](https://openreview.net/forum?id=8onaVSFTEj)
- [Faster Streaming and Scalable Algorithms for Finding Directed Dense Subgraphs in Large Graphs](https://openreview.net/forum?id=6h6ovHcC9G)
- [SiBBlInGS: Similarity-driven Building-Block Inference using Graphs across States](https://openreview.net/forum?id=h8aTi32tul)
- [Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching](https://openreview.net/forum?id=gE7qZurGH3)
- [Recurrent Distance Filtering for Graph Representation Learning](https://openreview.net/forum?id=5kGfm3Pa41)
- [Graph External Attention Enhanced Transformer](https://openreview.net/forum?id=0rV7VIrcjX)
- [A Graph is Worth $K$ Words: Euclideanizing Graph using Pure Transformer](https://openreview.net/forum?id=zxxSJAVQPc)
- [Convergence Guarantees for the DeepWalk Embedding on Block Models](https://openreview.net/forum?id=xwxUbBHC1q)
- [Delaunay Graph: Addressing Over-Squashing and Over-Smoothing Using Delaunay Triangulation](https://openreview.net/forum?id=uyhjKoaIQa)
- [Simulation of Graph Algorithms with Looped Transformers](https://openreview.net/forum?id=aA2326y3hf)
- [Efficient PAC Learnability of Dynamical Systems Over Multilayer Networks](https://openreview.net/forum?id=2PVjIQdq7N)
- [Gaussian Processes on Cellular Complexes](https://openreview.net/forum?id=afnyJfQddk)
- [An Efficient Maximal Ancestral Graph Listing Algorithm](https://openreview.net/forum?id=MZkqjV4FRT)
- [Extending Test-Time Augmentation with Metamorphic Relations for Combinatorial Problems](https://openreview.net/forum?id=PNsdnl8blk)
- [Graph Positional and Structural Encoder](https://openreview.net/forum?id=UTSCK582Yo)
- [Translating Subgraphs to Nodes Makes Simple GNNs Strong and Efficient for Subgraph Representation Learning](https://openreview.net/forum?id=xSizvCoI79)
- [Exploring Correlations of Self-Supervised Tasks for Graphs](https://openreview.net/forum?id=O3CFN1VIwt)
- [Surprisingly Strong Performance Prediction with Neural Graph Features](https://openreview.net/forum?id=EhPpZV6KLk)
- [Bipartite Matching in Massive Graphs: A Tight Analysis of EDCS](https://openreview.net/forum?id=EDEISRmi6X)
- [Uncertainty for Active Learning on Graphs](https://openreview.net/forum?id=BCEtumPYDt)
- [Learning Latent Structures in Network Games via Data-Dependent Gated-Prior Graph Variational Autoencoders](https://openreview.net/forum?id=kKWjZoaRLv)
- [Graph Neural Stochastic Diffusion for Estimating Uncertainty in Node Classification](https://openreview.net/forum?id=xJUhgvM2u8)
- [Position Paper: Future Directions in the Theory of Graph Machine Learning](https://openreview.net/forum?id=wBr5ozDEKp)
- [Pairwise Alignment Improves Graph Domain Adaptation](https://openreview.net/forum?id=ttnbM598vZ)
- [From Coarse to Fine: Enable Comprehensive Graph Self-supervised Learning with Multi-granular Semantic Ensemble](https://openreview.net/forum?id=JnA9IveEwg)
- [Identifiability Matters: Revealing the Hidden Recoverable Condition in Unbiased Learning to Rank](https://openreview.net/forum?id=JU3xHh1vWw)



<a name="PDE" />

## GNNs for PDE/ODE/Physics
- [Graph Neural PDE Solvers with Conservation and Similarity-Equivariance](https://openreview.net/forum?id=WajJf47TUi)
- [Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics](https://openreview.net/forum?id=Pbey7LqBRl)
- [Locality-Sensitive Hashing-Based Efficient Point Transformer with Applications in High-Energy Physics](https://openreview.net/forum?id=vJx6fld6l0)
- [PGODE: Towards High-quality System Dynamics Modeling](https://openreview.net/forum?id=jrE7geZekq)
- [Prometheus: Out-of-distribution Fluid Dynamics Modeling with Disentangled Graph ODE](https://openreview.net/forum?id=JsPvL6ExK8)
- [Equivariant Graph Neural Operator for Modeling 3D Dynamics](https://openreview.net/forum?id=dccRCYmL5x)
- [HAMLET: Graph Transformer Neural Operator for Partial Differential Equations](https://openreview.net/forum?id=nYX7I6PsL7)
- [Towards General Algorithm Discovery for Combinatorial Optimization: Learning Symbolic Branching Policy from Bipartite Graph](https://openreview.net/forum?id=ULleq1Dtaw)
- [PDHG-Unrolled Learning-to-Optimize Method for Large-Scale Linear Programming](https://openreview.net/forum?id=2cXzNDe614)
- [Combinatorial Approximations for Cluster Deletion: Simpler, Faster, and Better](https://openreview.net/forum?id=FpbKoIPHxb)

<a name="LLM" />

## Graph and Large Language Models/Agents
- [Graph-enhanced Large Language Models in Asynchronous Plan Reasoning](https://openreview.net/forum?id=eVGpdivOnQ)
- [GPTSwarm: Language Agents as Optimizable Graphs](https://openreview.net/forum?id=uTC9AFXIhg)
- [Case-Based or Rule-Based: How Do Transformers Do the Math?](https://openreview.net/forum?id=4Vqr8SRfyX)
- [SceneCraft: An LLM Agent for Synthesizing 3D Scenes as Blender Code](https://openreview.net/forum?id=gAyzjHw2ml)
- [LLaGA: Large Language and Graph Assistant](https://openreview.net/forum?id=B48Pzc4oKi)
- [MAGDi: Structured Distillation of Multi-Agent Interaction Graphs Improves Reasoning in Smaller Language Models](https://openreview.net/forum?id=ffLblkoCw8)
- [CHEMREASONER: Heuristic Search over a Large Language Models Knowledge Space using Quantum-Chemical Feedback](https://openreview.net/forum?id=3tJDnEszco)
- [Latent Logic Tree Extraction for Event Sequence Explanation from LLMs](https://openreview.net/forum?id=pwfcwEqdUz)


<a name="KG" />

## Knowledge Graph and Knowledge Graph Embeddings
- [Generalizing Knowledge Graph Embedding with Universal Orthogonal Parameterization](https://openreview.net/forum?id=Sv4u9PtvT5)
- [PAC-Bayesian Generalization Bounds for Knowledge Graph Representation Learning](https://openreview.net/forum?id=sOyJSNUrzQ)
- [Knowledge Graphs Can be Learned with Just Intersection Features](https://openreview.net/forum?id=Al5GlVytqi)
- [KnowFormer: Revisiting Transformers for Knowledge Graph Reasoning](https://openreview.net/forum?id=EncFNR3hxM)
- [Understanding Reasoning Ability of Language Models From the Perspective of Reasoning Paths Aggregation](https://openreview.net/forum?id=dZsEOFUDew)
- [Coarse-to-Fine Highlighting: Reducing Knowledge Hallucination in Large Language Models](https://openreview.net/forum?id=JCG0KTPVYy)
- [Knowledge-aware Reinforced Language Models for Protein Directed Evolution](https://openreview.net/forum?id=MikandLqtW)



<a name="SpatialTemporalGNNs" />

## Spatial and/or Temporal GNNs
- [Unlocking the Power of Spatial and Temporal Information in Medical Multimodal Pre-training](https://openreview.net/forum?id=87ZrVHDqmR)
- [S3GCL: Spectral, Swift, Spatial Graph Contrastive Learning](https://openreview.net/forum?id=znKAWRZSF9)
- [Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach](https://openreview.net/forum?id=UZlMXUGI6e)
- [Neurodegenerative Brain Network Classification via Adaptive Diffusion with Temporal Regularization](https://openreview.net/forum?id=GTnn6bNE3j)
- [Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning](https://openreview.net/forum?id=3FeYlKIPr3)
- [Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling](https://openreview.net/forum?id=uYIFQOtb58)
- [Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting](https://openreview.net/forum?id=nd47Za5jk5)
- [Long Range Propagation on Continuous-Time Dynamic Graphs](https://openreview.net/forum?id=gVg8V9isul)
- [Biharmonic Distance of Graphs and its Higher-Order Variants: Theoretical Properties with Applications to Centrality and Clustering](https://openreview.net/forum?id=3pxMIjB9QK)



<a name="apps" />

## GNN Applications
- [LaMAGIC: Language-Model-based Topology Generation for Analog Integrated Circuits](https://openreview.net/forum?id=MjGCD8wk1k)
- [SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity, ECG and Respiratory Signals](https://openreview.net/forum?id=QXqXGDapkQ)
- [Predicting and Interpreting Energy Barriers of Metallic Glasses with Graph Neural Networks](https://openreview.net/forum?id=7rTbqkKvA6)
- [CARTE: Pretraining and Transfer for Tabular Learning](https://openreview.net/forum?id=9kArQnKLDp)
- [Neurodegenerative Brain Network Classification via Adaptive Diffusion with Temporal Regularization](https://openreview.net/forum?id=GTnn6bNE3j)
- [A Circuit Domain Generalization Framework for Efficient Logic Synthesis in Chip Design](https://openreview.net/forum?id=1KemC8DNa0)
- [Towards an Understanding of Stepwise Inference in Transformers: A Synthetic Graph Navigation Model](https://openreview.net/forum?id=8VEGkphQaK)
- [Locality-Sensitive Hashing-Based Efficient Point Transformer with Applications in High-Energy Physics](https://openreview.net/forum?id=vJx6fld6l0)
- [Biharmonic Distance of Graphs and its Higher-Order Variants: Theoretical Properties with Applications to Centrality and Clustering](https://openreview.net/forum?id=3pxMIjB9QK)
- [Graph2Tac: Online Representation Learning of Formal Math Concepts](https://openreview.net/forum?id=A7CtiozznN)
- [The Merit of River Network Topology for Neural Flood Forecasting](https://openreview.net/forum?id=QE6iC9s6vU)
- [OT-CLIP: Understanding and Generalizing CLIP via Optimal Transport](https://openreview.net/forum?id=X8uQ1TslUc)
- [Bringing Motion Taxonomies to Continuous Domains via GPLVM on Hyperbolic manifolds](https://openreview.net/forum?id=ndVXXmxSC5)




<a name="xai" />

## Explainable AI
- [Structure Your Data: Towards Semantic Graph Counterfactuals](https://openreview.net/forum?id=OenMwDPqWn)
- [Temporal Spiking Neural Networks with Synaptic Delay for Graph Reasoning](https://openreview.net/forum?id=3FeYlKIPr3)
- [Federated Self-Explaining GNNs with Anti-shortcut Augmentations](https://openreview.net/forum?id=ZxDqSBgFSM)
- [Explaining Graph Neural Networks via Structure-aware Interaction Index](https://openreview.net/forum?id=2T00oYk54P)
- [Generating In-Distribution Proxy Graphs for Explaining Graph Neural Networks](https://openreview.net/forum?id=ohG9bVMs5j)
- [EiG-Search: Generating Edge-Induced Subgraphs for GNN Explanation in Linear Time](https://openreview.net/forum?id=HO0g6cHVZx)
- [Graph Neural Network Explanations are Fragile](https://openreview.net/forum?id=qIOSNyPPwB)



<a name="rl" />

## Reinforcement Learning
- [Tackling Non-Stationarity in Reinforcement Learning via Causal-Origin Representation](https://openreview.net/forum?id=WLGWMDtj8L)
- [SiT:   Symmetry-invariant Transformers for Generalisation in Reinforcement Learning](https://openreview.net/forum?id=SWrwurHAeq)
- [HGAP: Boosting Permutation Invariant and Permutation Equivariant in Multi-Agent Reinforcement Learning via Graph Attention Network](https://openreview.net/forum?id=KpUdNe9lsr)
- [Subequivariant Reinforcement Learning in 3D Multi-Entity Physical Environments](https://openreview.net/forum?id=hQpUhySEJi)
- [Breadth-First Exploration on Adaptive Grid for Reinforcement Learning](https://openreview.net/forum?id=59MYoLghyk)

<a name="molecular" />

## Graphs and Molecules
- [Swallowing the Bitter Pill: Simplified Scalable Conformer Generation](https://openreview.net/forum?id=I44Em5D5xy)
- [Representing Molecules as Random Walks Over Interpretable Grammars](https://openreview.net/forum?id=gS3nc9iUrH)
- [Expressivity and Generalization: Fragment-Biases for Molecular GNNs](https://openreview.net/forum?id=rPm5cKb1VB)
- [Generalist Equivariant Transformer Towards 3D Molecular Interaction Learning](https://openreview.net/forum?id=dWxb80a0TW)
- [UniCorn: A Unified Contrastive Learning Approach for Multi-view Molecular Representation Learning](https://openreview.net/forum?id=2NfpFwJfKu)
- [Modelling Microbial Communities with Graph Neural Networks](https://openreview.net/forum?id=vBJZ93tvoE)
- [Structure-Aware E(3)-Invariant Molecular Conformer Aggregation Networks](https://openreview.net/forum?id=qGEEso256L)
- [Projecting Molecules into Synthesizable Chemical Spaces](https://openreview.net/forum?id=scFlbJQdm1)
- [Proteus: Exploring Protein Structure Generation for Enhanced Designability and Efficiency](https://openreview.net/forum?id=IckJCzsGVS)
- [CHEMREASONER: Heuristic Search over a Large Language Models Knowledge Space using Quantum-Chemical Feedback](https://openreview.net/forum?id=3tJDnEszco)


<a name="GFlowNets" />

## **GFlowNets**
- [Latent Logic Tree Extraction for Event Sequence Explanation from LLMs](https://openreview.net/forum?id=pwfcwEqdUz)
- [GFlowNet Training by Policy Gradients](https://openreview.net/forum?id=G1igwiBBUj)
- [Embarrassingly Parallel GFlowNets](https://openreview.net/forum?id=KJhLpzqNri)
- [Learning to Scale Logits for Temperature-Conditional GFlowNets](https://openreview.net/forum?id=GUEsK9xJny)



<a name="casual" />

## Casual Discovery and Graphs
- [Causal-IQA: Towards the Generalization of Image Quality Assessment Based on Causal Inference](https://openreview.net/forum?id=gKPkipJ3gm)
- [Optimal Transport for Structure Learning Under Missing Data](https://openreview.net/forum?id=09Robz3Ppy)
- [Causal Representation Learning from Multiple Distributions: A General Setting](https://openreview.net/forum?id=Pte6iiXvpf)
- [Foundations of Testing for Finite-Sample Causal Discovery](https://openreview.net/forum?id=oUmXcewb83)
- [Optimal Kernel Choice for Score Function-based Causal Discovery](https://openreview.net/forum?id=DYd4vyyhUu)
- [Causal Discovery with Fewer Conditional Independence Tests](https://openreview.net/forum?id=HpT19AKddu)
- [How Transformers Learn Causal Structure with Gradient Descent](https://openreview.net/forum?id=jNM4imlHZv)
- [From Geometry to Causality- Ricci Curvature and the Reliability of Causal Inference on Networks](https://openreview.net/forum?id=4DAl3IsvlU)
- [A Fixed-Point Approach for Causal Generative Modeling](https://openreview.net/forum?id=JpzIGzru5F)
- [Neural Tangent Kernels Motivate Cross-Covariance Graphs in Neural Networks](https://openreview.net/forum?id=61JD8wp4Id)
- [Scalable and Flexible Causal Discovery with an Efficient Test for Adjacency](https://openreview.net/forum?id=5M4Qa9AqY7)
- [Discovering Mixtures of Structural Causal Models from Time Series Data](https://openreview.net/forum?id=cHJAUdam3i)
- [Challenges and Considerations in the Evaluation of Bayesian Causal Discovery](https://openreview.net/forum?id=bqgtkBDkNs)
- [Causal Effect Identification in LiNGAM Models with Latent Confounders](https://openreview.net/forum?id=C1iNBLIClt)
- [Adaptive Online Experimental Design for Causal Discovery](https://openreview.net/forum?id=nJzf3TVnOn)
- [Stable Differentiable Causal Discovery](https://openreview.net/forum?id=JJZBZW28Gn)


<a name="FL" />

## Federated Learning, Privacy, Decentralization
- [Federated Self-Explaining GNNs with Anti-shortcut Augmentations](https://openreview.net/forum?id=ZxDqSBgFSM)
- [Beyond the Federation: Topology-aware Federated Learning for Generalization to Unseen Clients](https://openreview.net/forum?id=2zLt2Odckx)
- [Effective Federated Graph Matching](https://openreview.net/forum?id=rSfzchjIYu)
- [The Privacy Power of Correlated Noise in Decentralized Learning](https://openreview.net/forum?id=5JrlywYHRi)
- [Privacy Attacks in Decentralized Learning](https://openreview.net/forum?id=mggc3oYHy4)
- [Differentially Private Decentralized Learning with Random Walks](https://openreview.net/forum?id=k2dVVIWWho)


<a name="SceneGraphs" />

## Scene Graphs
- [SceneCraft: An LLM Agent for Synthesizing 3D Scenes as Blender Code](https://openreview.net/forum?id=gAyzjHw2ml)
- [Scene Graph Generation Strategy with Co-occurrence Knowledge and Learnable Term Frequency](https://openreview.net/forum?id=tTq3qMkJ8w)
- [Video-of-Thought: Step-by-Step Video Reasoning from Perception to Cognition](https://openreview.net/forum?id=fO31YAyNbI)



<a name="PositionPapers" />

# Position Papers
- [Position: Topological Deep Learning is the New Frontier for Relational Learning](https://openreview.net/forum?id=Nl3RG5XWAt)
- [Position: Graph Foundation Models Are Already Here](https://openreview.net/forum?id=Edz0QXKKAo)
- [Position Paper: Future Directions in the Theory of Graph Machine Learning](https://openreview.net/forum?id=wBr5ozDEKp)
- [Position: Relational Deep Learning - Graph Representation Learning on Relational Databases](https://openreview.net/forum?id=BIMSHniyCP)




# Others
- [Exploring the Enigma of Neural Dynamics Through A Scattering-Transform Mixer Landscape for Riemannian Manifold](https://openreview.net/forum?id=EYOo48YGhy)
- [Open Ad Hoc Teamwork with Cooperative Game Theory](https://openreview.net/forum?id=RlibRvH4B4)
- [Incorporating Information into Shapley Values: Reweighting via a Maximum Entropy Approach](https://openreview.net/forum?id=DwniHlwcOB)
- [Predicting Lagrangian Multipliers for Mixed Integer Linear Programs](https://openreview.net/forum?id=aZnZOqUOHq)
- [MAGNOLIA: Matching Algorithms via GNNs for Online Value-to-go Approximation](https://openreview.net/forum?id=XlgeQ47Ra9)
- [Dynamic Metric Embedding into lp Space](https://openreview.net/forum?id=z3PUNzdmGs)
- [Graph Mixup on Approximate GromovWasserstein Geodesics](https://openreview.net/forum?id=PKdege0U6Z)
- [Differentiability and Optimization of Multiparameter Persistent Homology](https://openreview.net/forum?id=ixdfvnO0uy)
- [Graph-Triggered Rising Bandits](https://openreview.net/forum?id=bPsohGR6gD)
- [On Interpolating Experts and Multi-Armed Bandits](https://openreview.net/forum?id=qIiPM5CbRY)
- [Multiplicative Weights Update, Area Convexity and Random Coordinate Descent for Densest Subgraph Problems](https://openreview.net/forum?id=d2E2i5rJ4x)
- [Empowering Graph Invariance Learning with Deep Spurious Infomax](https://openreview.net/forum?id=u9oSQtujCF)
- [Rethinking Independent Cross-Entropy Loss For Graph-Structured Data](https://openreview.net/forum?id=zrQIc9mQQN)
- [Predictive Coding beyond Correlations](https://openreview.net/forum?id=nTgzmXvuEA)
- [When is Transfer Learning Possible?](https://openreview.net/forum?id=9yADTDHgGu)
- [Optimal Exact Recovery in Semi-Supervised Learning: A Study of Spectral Methods and Graph Convolutional Networks](https://openreview.net/forum?id=8m4V6Fx6ma)
- [Feature Distribution on Graph Topology Mediates the Effect of Graph Convolution: Homophily Perspective](https://openreview.net/forum?id=buW1Bi6XFw)
- [Graph As Point Set](https://openreview.net/forum?id=b6yHkQpSwZ)
- [CKGConv: General Graph Convolution with Continuous Kernels](https://openreview.net/forum?id=KgfGxXbjjE)
- [REST: Efficient and Accelerated EEG Seizure Analysis through Residual State Updates](https://openreview.net/forum?id=9GbAea74O6)
- [VisionGraph: Leveraging Large Multimodal Models for Graph Theory Problems in Visual Context](https://openreview.net/forum?id=gjoUXwuZdy)
- [Finite Volume Features, Global Geometry Representations, and Residual Training for Deep Learning-based CFD Simulation](https://openreview.net/forum?id=WzD4a5ufN8)
- [Two Heads Are Better Than One: Boosting Graph Sparse Training via Semantic and Topological Awareness](https://openreview.net/forum?id=WDgV1BJEW0)
- [DUPLEX: Dual GAT for Complex Embedding of Directed Graphs](https://openreview.net/forum?id=M3uv4qDKOL)
- [MS-TIP: Imputation Aware Pedestrian Trajectory Prediction](https://openreview.net/forum?id=s4Hy0L4mml)
- [Learning Graph Representation via Graph Entropy Maximization](https://openreview.net/forum?id=xwOENWCo46)
- [QBMK: Quantum-based Matching Kernels for Un-attributed Graphs](https://openreview.net/forum?id=PYDCwWvbG7)
- [Mitigating Label Noise on Graphs via Topological Sample Selection](https://openreview.net/forum?id=BRIcZiK5Fr)
- [Multi-View Stochastic Block Models](https://openreview.net/forum?id=BJx1K4lAAX)
- [Learning in Deep Factor Graphs with Gaussian Belief Propagation](https://openreview.net/forum?id=6WYk5R86Wl)
- [Individual Fairness in Graph Decomposition](https://openreview.net/forum?id=8f8SI9X9ox)
- [Graphon Mean Field Games with a Representative Player: Analysis and Learning Algorithm](https://openreview.net/forum?id=7C4EQqtb02)
- [Sign Rank Limitations for Inner Product Graph Decoders](https://openreview.net/forum?id=Lb8G2dZjcB)
- [Differentiable Mapper for Topological Optimization of Data Representation](https://openreview.net/forum?id=QZ1DVzr6N9)
- [Incremental Topological Ordering and Cycle Detection with Predictions](https://openreview.net/forum?id=wea7nsJdMc)


---


**Missing any paper?**
If any paper is absent from the list, please feel free to [mail](mailto:azminetoushik.wasi@gmail.com) or [open an issue](https://github.com/azminewasi/Awesome-Graph-Research-ICML2024/issues/new/choose) or submit a pull request. I'll gladly add that! Also, If I mis-categorized, please knock!

---

## More Collectons:
- [**Awesome ICLR 2024 Graph Paper Collection**](https://github.com/azminewasi/Awesome-Graph-Research-ICLR2024)
- [**Awesome-LLMs-ICLR-24**](https://github.com/azminewasi/Awesome-LLMs-ICLR-24/)

---

## ✨ **Credits**
**Azmine Toushik Wasi**

 [![website](https://img.shields.io/badge/-Website-blue?style=flat-square&logo=rss&color=1f1f15)](https://azminewasi.github.io) 
 [![linkedin](https://img.shields.io/badge/LinkedIn-%320beff?style=flat-square&logo=linkedin&color=1f1f18)](https://www.linkedin.com/in/azmine-toushik-wasi/) 
 [![kaggle](https://img.shields.io/badge/Kaggle-%2320beff?style=flat-square&logo=kaggle&color=1f1f1f)](https://www.kaggle.com/azminetoushikwasi) 
 [![google-scholar](https://img.shields.io/badge/Google%20Scholar-%2320beff?style=flat-square&logo=google-scholar&color=1f1f18)](https://scholar.google.com/citations?user=X3gRvogAAAAJ&hl=en) 
 [![facebook](https://img.shields.io/badge/Facebook-%2320beff?style=flat-square&logo=facebook&color=1f1f15)](https://www.facebook.com/cholche.gari.zatrabari/)
