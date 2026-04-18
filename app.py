import streamlit as st
import os
import tempfile
import logging
import sqlite3
import hashlib
import re
from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    st.error("请在 .env 文件中设置 ZHIPUAI_API_KEY")
    st.stop()

st.set_page_config(page_title="企业内部智能助理")
st.title("🏢 企业内部智能助理（多Agent协作）")

# ---------- SQLite 数据库 ----------
DB_PATH = "conversations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  question TEXT,
                  intent TEXT,
                  answer TEXT,
                  timestamp TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(question: str, intent: str, answer: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history (question, intent, answer, timestamp) VALUES (?, ?, ?, ?)",
              (question, intent, answer, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_recent_history(limit: int = 10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT question, intent, answer, timestamp FROM history ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def clear_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"清空历史失败: {e}")
        return False

init_db()

# ---------- 内存缓存 ----------
cache = {}

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.encode()).hexdigest()

# ---------- 加载 RAG 链 ----------
@st.cache_resource
def load_rag_chain():
    docs_path = "policy.pdf"
    if not os.path.exists(docs_path):
        with open("policy.txt", "w", encoding="utf-8") as f:
            f.write("公司年假政策：入职满1年享5天，满3年享10天。加班政策：工作日加班1.5倍工资，周末2倍。")
        docs_path = "policy.txt"

    if docs_path.endswith(".txt"):
        loader = TextLoader(docs_path, encoding="utf-8")
    elif docs_path.endswith(".pdf"):
        loader = PyPDFLoader(docs_path)
    else:
        raise ValueError("不支持的文件格式")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embeddings = ZhipuAIEmbeddings(zhipuai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    llm = ChatZhipuAI(model="glm-4-flash", zhipuai_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

# ---------- 模拟工具 ----------
def mock_leave_apply(days: int = 3, leave_type: str = "年假") -> str:
    return f"已提交{leave_type}申请，共{days}天，等待审批。"

def mock_reimbursement_query(expense_id: str = "RE2024001") -> str:
    return f"报销单号 {expense_id} 状态：财务审核中，预计3个工作日内到账。"

def mock_overtime_query() -> str:
    return "加班政策：工作日加班1.5倍工资，周末加班2倍，需提前在OA系统申请。"

def mock_complaint_suggestion(content: str) -> str:
    return f"感谢您的反馈，我们已经记录：{content}，会尽快处理。"

def mock_it_support(issue: str) -> str:
    if "密码" in issue:
        return "请通过企业微信联系IT部门重置密码。"
    elif "电脑" in issue or "卡" in issue:
        return "建议重启电脑或联系IT支持。"
    else:
        return "请联系IT部门获取帮助。"

# ---------- 状态 ----------
class AgentState(TypedDict):
    question: str
    intent: str
    answer: str

# ---------- 意图识别 ----------
def classify_intent(state: AgentState) -> AgentState:
    q = state["question"].lower()
    if any(kw in q for kw in ["请假", "年假", "调休", "事假", "病假"]):
        intent = "leave"
    elif any(kw in q for kw in ["报销", "费用", "发票"]):
        intent = "reimbursement"
    elif any(kw in q for kw in ["加班", "overtime"]):
        intent = "overtime"
    elif any(kw in q for kw in ["投诉", "建议", "意见", "反馈"]):
        intent = "complaint"
    elif any(kw in q for kw in ["政策", "规定", "制度", "年假怎么", "加班怎么"]):
        intent = "policy"
    else:
        intent = "it"
    logging.info(f"Intent: {intent}")
    state["intent"] = intent
    return state

# ---------- 工具节点 ----------
def leave_node(state: AgentState) -> AgentState:
    days_match = re.search(r"(\d+)天", state["question"])
    days = int(days_match.group(1)) if days_match else 3
    state["answer"] = mock_leave_apply(days)
    return state

def reimbursement_node(state: AgentState) -> AgentState:
    state["answer"] = mock_reimbursement_query()
    return state

def overtime_node(state: AgentState) -> AgentState:
    state["answer"] = mock_overtime_query()
    return state

def complaint_node(state: AgentState) -> AgentState:
    state["answer"] = mock_complaint_suggestion(state["question"])
    return state

def policy_node(state: AgentState) -> AgentState:
    qa_chain = load_rag_chain()
    key = get_cache_key(state["question"])
    if key in cache:
        answer = cache[key]
        logging.info("Cache hit")
    else:
        answer = qa_chain.invoke(state["question"])
        cache[key] = answer
        logging.info("Cache miss, stored")
    state["answer"] = answer
    return state

def it_node(state: AgentState) -> AgentState:
    state["answer"] = mock_it_support(state["question"])
    return state

# ---------- 路由 ----------
def route_intent(state: AgentState) -> Literal["leave", "reimbursement", "overtime", "complaint", "policy", "it"]:
    return state["intent"]

# ---------- 构建图 ----------
builder = StateGraph(AgentState)
builder.add_node("classify", classify_intent)
builder.add_node("leave", leave_node)
builder.add_node("reimbursement", reimbursement_node)
builder.add_node("overtime", overtime_node)
builder.add_node("complaint", complaint_node)
builder.add_node("policy", policy_node)
builder.add_node("it", it_node)

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", route_intent, {
    "leave": "leave",
    "reimbursement": "reimbursement",
    "overtime": "overtime",
    "complaint": "complaint",
    "policy": "policy",
    "it": "it"
})
for node in ["leave", "reimbursement", "overtime", "complaint", "policy", "it"]:
    builder.add_edge(node, END)

graph = builder.compile()

# ---------- Streamlit UI ----------
st.markdown("### 智能助理功能")
st.write("支持：请假申请、报销查询、加班政策、投诉建议、政策问答、IT支持")

# 侧边栏：文件上传
uploaded_file = st.sidebar.file_uploader("上传政策文档（TXT/PDF）", type=["txt", "pdf"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    st.sidebar.success("文档已上传，将用于政策问答（需重启生效）")

# 侧边栏：历史记录和清空按钮
st.sidebar.header("📜 最近对话")
# 使用 session_state 确保清空操作只执行一次
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False

if st.sidebar.button("清除所有历史记录"):
    st.session_state.clear_flag = True

if st.session_state.clear_flag:
    with st.sidebar.popover("确认清除"):
        st.warning("确定要清除所有历史记录吗？此操作不可撤销。")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认清除", type="primary"):
                if clear_history():
                    st.success("历史记录已清除！")
                    st.session_state.clear_flag = False
                    st.rerun()
                else:
                    st.error("清除失败，请检查数据库权限")
                    st.session_state.clear_flag = False
                    st.rerun()
        with col2:
            if st.button("❌ 取消"):
                st.info("已取消")
                st.session_state.clear_flag = False
                st.rerun()

history = get_recent_history(5)
if history:
    for q, intent, a, ts in history:
        st.sidebar.text(f"Q: {q[:50]}...")
        st.sidebar.text(f"意图: {intent} | 时间: {ts[:16]}")
        st.sidebar.divider()
else:
    st.sidebar.info("暂无历史记录")

# 主界面提问
question = st.text_input("请输入您的问题：")
if question:
    with st.spinner("思考中..."):
        initial_state = {"question": question, "intent": "", "answer": ""}
        final_state = graph.invoke(initial_state)
        answer = final_state["answer"]
        intent = final_state["intent"]
        st.write("**回答：**", answer)
        save_to_db(question, intent, answer)