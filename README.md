# 简单 RAG 实现指南

## 简介
本文档旨在指导用户如何简单地实现 RAG（Retrieval-Augmented Generation）模型，该模型结合了检索和生成步骤来提升自然语言处理任务的效果。通过利用现有的大规模预训练模型，本方法通过检索相关内容来增强生成的文本，从而改善模型对于各种自然语言任务的处理能力。

## 模型下载
在实现 RAG 模型之前，您需要下载以下两个模型文件：

### 1. Embedding 模型
用于生成句子或段落的嵌入向量，这些向量将用于支持模型的检索过程。
- **下载地址**: [BAAI/bge-large-zh-v1.5](https://huggingface.co/BAAI/bge-large-zh-v1.5/tree/main)

### 2. qwen0.5b model
这是一个基于 Transformer 架构的生成模型，专注于生成自然语言文本。
- **下载地址**: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
