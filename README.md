# Algorithms in Python — Article Series

A comprehensive series of articles exploring machine learning algorithms, each with
clear explanations and Python implementations.

## Workflow

1. **Learn** — Work through each algorithm together, building understanding
2. **Write individual articles** — One article per algorithm (`article.md` + `code.py`)
3. **Write category overviews** — One summary article per category (e.g., "Supervised Learning")
4. **Write master overview** — One article tying the entire series together

## Series Structure

### 00 — Foundations: Data Structures for AI
| # | Topic | Why it matters | Status |
|---|-------|----------------|--------|
| 1 | Arrays | The basic container — NumPy arrays underpin all numerical computation | ✅ |
| 2 | Matrices | Linear algebra's workhorse — weight matrices, transformations, covariance | ✅ |
| 3 | Tensors | Generalised matrices — the data format of deep learning (PyTorch, TensorFlow) | ✅ |
| 4 | Linked Lists | Sequential access — used in memory management and streaming data pipelines | ✅ |
| 5 | Graphs | Nodes and edges — social networks, knowledge bases, GNNs, spectral clustering | ✅ |
| 6 | Hash Tables | O(1) lookup — feature hashing, caching, deduplication, count-min sketches | ✅ |
| 7 | Queues | FIFO ordering — BFS, experience replay in RL, message passing, job scheduling | ✅ |
| 8 | Trees | Hierarchical structure — decision trees, random forests, KD-trees, B-trees | ✅ |
| 9 | Knowledge Graphs | Structured relational data — entity embeddings, reasoning, RAG systems | ✅ |
| 10 | Probabilistic Data Structures | Bloom filters, Count-Min Sketch, HyperLogLog — sublinear memory at the cost of bounded error | ✅ |
| 11 | Sparse Matrices | CSR / CSC / COO — TF-IDF, recommender systems, GNN adjacency, sparse autoencoders | ✅ |
| 12 | Vector Indexes (ANN) | HNSW, IVF, PQ, LSH — vector databases, RAG, FAISS / Pinecone / Chroma | ✅ |

### 01 — Supervised Learning
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Linear Regression | ✅ |
| 2 | Logistic Regression | ✅ |
| 3 | Naive Bayes | ✅ |
| 4 | K-Nearest Neighbours | ⬜ |
| 5 | Decision Trees | ✅ |

### 02 — Advanced Supervised Learning
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Random Forests | ✅ |
| 2 | Gradient Boosting (XGBoost / LightGBM / CatBoost) | ✅ |
| 3 | Support Vector Machines | ✅ |

### 03 — Unsupervised Learning
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | K-Means Clustering | ✅ |
| 2 | Hierarchical Clustering | ✅ |
| 3 | Principal Component Analysis (PCA) | ✅ |
| 4 | t-SNE | ✅ |
| 5 | UMAP | ✅ |
| 6 | Non-negative Matrix Factorisation (NMF) | ⬜ |
| 7 | Spectral Clustering | ⬜ |
| 8 | Association Rule Mining | ⬜ |

### 04 — Advanced Unsupervised Learning
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | DBSCAN | ⬜ |
| 2 | Gaussian Mixture Models | ⬜ |
| 3 | Autoencoders | ⬜ |
| 4 | Anomaly Detection | ⬜ |
| 5 | Latent Dirichlet Allocation (LDA) | ⬜ |

### 05 — Bayesian, Probabilistic & Causal Methods
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Gaussian Processes | ⬜ |
| 2 | Markov Chain Monte Carlo (MCMC) | ⬜ |
| 3 | Variational Inference | ⬜ |
| 4 | Probabilistic Programming | ⬜ |
| 5 | Causal Inference | ⬜ |

### 06 — Time Series & Forecasting
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | ARIMA | ⬜ |
| 2 | Exponential Smoothing | ⬜ |
| 3 | State-Space Models | ⬜ |
| 4 | Prophet | ⬜ |
| 5 | Temporal Fusion Transformer (TFT) | ⬜ |

### 07 — Recommender Systems
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Matrix Factorisation (SVD / ALS) | ⬜ |
| 2 | Neural Collaborative Filtering | ⬜ |
| 3 | Two-Tower Retrieval | ⬜ |
| 4 | Sequential Recommenders | ⬜ |

### 08 — Reinforcement Learning
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Q-Learning | ⬜ |
| 2 | SARSA | ⬜ |
| 3 | Deep Q-Networks (DQN) | ⬜ |
| 4 | Policy Gradient Methods | ⬜ |
| 5 | Advantage Actor-Critic (A2C / A3C) | ⬜ |

### 09 — Advanced Reinforcement Learning
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Proximal Policy Optimisation (PPO) | ⬜ |
| 2 | Trust Region Policy Optimisation (TRPO) | ⬜ |
| 3 | Deep Deterministic Policy Gradient (DDPG) | ⬜ |
| 4 | Twin Delayed DDPG (TD3) | ⬜ |
| 5 | Soft Actor-Critic (SAC) | ⬜ |
| 6 | Monte Carlo Tree Search (MCTS) | ⬜ |
| 7 | Offline Reinforcement Learning | ⬜ |

### 10 — Semi-Supervised Learning
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Self-Training | ⬜ |
| 2 | Co-Training | ⬜ |
| 3 | Multiview Learning | ⬜ |
| 4 | Expectation Maximisation (EM) | ⬜ |
| 5 | Label Propagation | ⬜ |
| 6 | Graph-Based Methods | ⬜ |
| 7 | Tri-Training | ⬜ |
| 8 | Virtual Adversarial Training (VAT) | ⬜ |
| 9 | Deep Generative Models for SSL | ⬜ |
| 10 | Transductive SVM | ⬜ |
| 11 | Active Learning | ⬜ |

### 11 — Deep Learning Architectures
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Multi-Layer Perceptron (MLP) | ⬜ |
| 2 | Convolutional Neural Networks (CNNs) | ⬜ |
| 3 | Recurrent Neural Networks (RNN / LSTM / GRU) | ⬜ |
| 4 | Attention Mechanisms | ⬜ |
| 5 | Transformers | ⬜ |
| 6 | Graph Neural Networks (GCN / GAT / GraphSAGE) | ⬜ |
| 7 | Generative Adversarial Networks (GANs) | ⬜ |
| 8 | Variational Autoencoders (VAEs) | ⬜ |
| 9 | Diffusion Models | ⬜ |
| 10 | Normalising Flows | ⬜ |

### 12 — Self-Supervised Learning
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Contrastive Learning (SimCLR / CLIP) | ⬜ |
| 2 | Masked Autoencoding (MAE / BERT-style) | ⬜ |
| 3 | Self-Distillation (DINO) | ⬜ |
| 4 | Momentum Contrast (MoCo) | ⬜ |

### 13 — Natural Language Processing
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Word Embeddings (Word2Vec / GloVe / fastText) | ⬜ |
| 2 | BPE / WordPiece / SentencePiece Tokenisation | ⬜ |
| 3 | Sequence-to-Sequence Models | ⬜ |
| 4 | BERT-style Encoders | ⬜ |
| 5 | GPT-style Decoders | ⬜ |
| 6 | Natural Language Inference (NLI) | ⬜ |
| 7 | Natural Language Generation (NLG) | ⬜ |
| 8 | Retrieval-Augmented Generation (RAG) | ⬜ |
| 9 | Fine-tuning & LoRA | ⬜ |
| 10 | RLHF / DPO | ⬜ |
| 11 | In-Context Learning & Prompting | ⬜ |

### 14 — Computer Vision
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | Image Classification (ResNet / EfficientNet / ViT) | ⬜ |
| 2 | Object Detection (YOLO / R-CNN family) | ⬜ |
| 3 | Semantic Segmentation (U-Net) | ⬜ |
| 4 | Generative Vision (StyleGAN / Stable Diffusion) | ⬜ |
| 5 | Spiking Neural Networks | ⬜ |

### 15 — Large-Scale & Production
| # | Algorithm | Status |
|---|-----------|--------|
| 1 | MapReduce | ⬜ |
| 2 | Distributed Training | ⬜ |
| 3 | Federated Learning | ⬜ |
| 4 | Graph Processing Algorithms | ⬜ |
| 5 | Large-Scale Optimisation | ⬜ |

### Series Overviews
| Article | Status |
|---------|--------|
| Foundations: Data Structures Overview | ⬜ |
| Supervised Learning Overview | ⬜ |
| Advanced Supervised Learning Overview | ⬜ |
| Unsupervised Learning Overview | ⬜ |
| Advanced Unsupervised Learning Overview | ⬜ |
| Bayesian, Probabilistic & Causal Methods Overview | ⬜ |
| Time Series & Forecasting Overview | ⬜ |
| Recommender Systems Overview | ⬜ |
| Reinforcement Learning Overview | ⬜ |
| Advanced Reinforcement Learning Overview | ⬜ |
| Semi-Supervised Learning Overview | ⬜ |
| Deep Learning Architectures Overview | ⬜ |
| Self-Supervised Learning Overview | ⬜ |
| Natural Language Processing Overview | ⬜ |
| Computer Vision Overview | ⬜ |
| Large-Scale & Production Overview | ⬜ |
| **Master Overview — All Algorithms** | ⬜ |

---

**Total: 105 articles (12 foundations + 93 algorithms) + 16 category overviews + 1 master overview = 122 articles**

### Deferred / out of scope
- **Quantum Machine Learning** — research-grade and a distinct field; planned as a future stand-alone series rather than tacked on here.
