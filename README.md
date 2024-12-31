## Canvas # Text Autocompletion with RAG

一个智能文本自动补全系统，采用GPT-2或其他开源LLM，并与检索加强生成技术(RAG)相结合，提供与上下文相关的文本提示。

## 🚀 Features / 特点

- **Smart Autocompletion / 智能自动补全**: 生成与上下文相关的文本连续
- **RAG Enhancement / RAG增强**: 利用外部文档，提升提示质量
- **FastAPI Backend / FastAPI后端**: 高性能的REST API用于文本生成
- **Streamlit Frontend / Streamlit前端**: 便捷用户界面，便于测试
- **Evaluation Tools / 评价工具**: 内置模型性能评价指标

## 🛠️ Quick Start / 快速开始

### Installation / 安装

```bash
# 克隆仓库
克隆仓库
git clone https://github.com/LJXjean/Text-autocompletion
cd Text-autocompletion

# 安装依赖
pip install -r requirements.txt
```

### Running the Application / 运行应用

1. **Start the Backend Server / 启动后端服务器**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

2. **Launch the Frontend Demo / 启动前端演示**
```bash
streamlit run frontend_demo.py
```

3. **Run Evaluation (Optional) / 运行评价(可选)**
```bash
python eval.py
```

4. **查阅设计方案和评估方案分别在 design_plan.md 和 evaluation.md**

## 🗰️ System Architecture / 系统架构

```
Frontend (Streamlit) ─▶ Server (FastAPI) ─├─▶ RAG (FAISS + Embeddings)
                                         └─▶ AutocompleteModel (GPT-2)
```

## 💡 How It Works / 工作原理

1. **Input Processing / 输入处理**: 系统从用户的光标位置接收文本
2. **Document Retrieval / 文档检索**: RAG组件从文档库中检索相关信息
3. **Text Generation / 文本生成**: 模型结合用户输入和检索信息，生成补全文本
4. **Response / 返回**: 返回与上下文相关的文本提示

## 技术路线与整体架构

###  技术栈选型

1. **LLM**：使用 Hugging Face 提供的 **GPT2** 预训练模型，或基于 GPT2 做小规模微调。
2. **检索**：基于 **LangChain** + **FAISS** 实现向量检索。
3. **后端框架**：**FastAPI** 提供 RESTful API。
4. **前端 Demo**：可使用 **Streamlit** 实现一个最小可行的编辑器界面。

## **实现细节**

数据：[RealTimeData/bbc_news_alltime](https://huggingface.co/datasets/RealTimeData/bbc_news_alltime)
将数据清洗和切分后（context 和 continuation）存储为 **JSONL** 格式，并在“训练集”和“测试/评测集”之间做好划分。
同时抽取部分bbc_news作为rag的documents作为外部引用

### 模型微调finetuning（future feature）
---
- 使用 **Trainer** + **GPT2LMHeadModel** 进行简单的微调，Epoch = 1~2，Batch Size 适当设置；
- 如果时间或算力有限，可直接使用预训练 GPT2 进行推理，不做微调。

### 检索增强（RAG）
---
- 在 **rag_utils.py** 中，采用 **LangChain** 提供的 **FAISS** 向量索引：
    1. 遍历 JSONL 文档，每行（或按照 chunk）提取文本字段（如 `content`），
    2. 使用 **HuggingFaceEmbeddings**（MiniLM 等）生成向量；
    3. 建立 FAISS 索引并提供 `search(query, top_k)` 接口；
- 生成时，调用 `rag_helper.search(user_context, top_k=3)` 获得最相近的文本片段，拼接到 Prompt 里。

### 服务器
---
在后端中将模型生成文本函数封装成一个api端口 并在端口内使用rag来帮助：

- 接收 `text_before_cursor`；
- 使用 `rag_helper.search` 检索 doc 片段；
- 调用 `autocomplete_model.generate_text` 进行推理；

### 文本自动补全模型（AutocompleteModel）
---
用于封装 **GPT2** 推理逻辑 或 调用自己部署并微调后的模型给服务器endpoint提供服务

###  前端 Demo
---
- 使用 **Streamlit**，在 `frontend_demo.py` 中提供简单的文本框输入 + “自动补全”按钮，调用后端接口显示结果。

## 🔧 Configuration (Future Feature) / 配置(未来功能)

可在`config.yaml`中调整主要参数：
- 模型选择
- RAG设置（切块大小，检索数量）
- 生成参数

## 📚 API Documentation / API文档

### Autocomplete Endpoint / 自动补全端点

```bash
POST /autocomplete
{
    "text_before_cursor": "Your input text here"
}
```

