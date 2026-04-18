# 企业内部智能助理（多Agent协作）

## 项目简介
基于 LangGraph 开发的多智能体企业内部助理系统，支持请假申请、报销查询、加班政策、投诉建议、政策问答、IT支持等六大功能。系统采用多Agent协作架构，实现意图识别、工具调用、状态管理，并集成 RAG 知识库、SQLite 历史记录、内存缓存等工程化组件。

## 技术栈
- LangGraph：多Agent工作流编排
- LangChain：RAG 知识库检索
- FAISS：向量存储
- SQLite：对话历史存储
- Streamlit：前端演示界面
- FastAPI：API 接口封装
- Docker：容器化部署
- 智谱 AI API：大模型调用

## 快速开始

### 本地运行
1. 克隆仓库
2. 安装依赖：`pip install -r requirements.txt`
3. 创建 `.env` 文件，添加 `ZHIPUAI_API_KEY=你的密钥`
4. 运行：`streamlit run app.py`

### Docker 运行
```bash
docker build -t employee-assistant .
docker run -p 8501:8501 employee-assistant