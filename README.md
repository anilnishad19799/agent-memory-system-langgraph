# Agent Memory System with LangGraph 🧠

A comprehensive implementation of **Short-Term Memory (STM)** and **Long-Term Memory (LTM)** for LLM-powered agents using LangGraph, PostgreSQL, and pgvector for intelligent semantic search.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0+-green.svg)](https://docs.langchain.com/oss/python/langgraph)

---

## 🎯 Overview

This project demonstrates how to build AI agents that actually remember things. Unlike traditional chatbots that start fresh with each conversation, this system implements:

- **Short-Term Memory (STM)**: Conversation history within a thread with PostgreSQL persistence
- **Long-Term Memory (LTM)**: User facts and preferences that persist across conversations with semantic search
- **Complete Agent Examples**: Real-world scenarios combining both memory types

```
User Message
    ↓
[Load STM] ← Previous messages from this conversation
[Load LTM] ← User facts from all previous conversations
    ↓
[Agent Processes with Full Context]
    ↓
[Update STM] → Save to checkpointer
[Update LTM] → Save facts to vector store
    ↓
Response
```

---

## 📚 Contents

### Notebooks Included

| File | Description | Purpose |
|------|-------------|---------|
| **stm_basic_inmemorysaver.ipynb** | STM with in-memory storage | Development & understanding |
| **stm_persistence.ipynb** | STM with PostgreSQL persistence | Production-ready conversation memory |
| **ltm_basic_memorystore.ipynb** | LTM with in-memory vector store | Development & understanding |
| **ltm_persistence.ipynb** | LTM with PostgreSQL + pgvector | Production-ready semantic search |
| **stm_ltm_combined_agent.ipynb** | Complete agent using both STM & LTM | Real-world example: Finance Assistant |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 15+ (or Docker)
- API Keys: OpenAI

### 1. Install Dependencies

```bash
pip install langgraph langchain langchain-anthropic langchain-openai
pip install langgraph-checkpoint-postgres langgraph-postgres psycopg[binary,pool]
pip install python-dotenv jupyter
```

### 2. Setup PostgreSQL with pgvector

#### Option A: Using Docker (Recommended)
```bash
# Pull pgvector-enabled PostgreSQL
docker pull ankane/pgvector

# Run container
docker run --name langgraph_db \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=agent_memory \
  -p 5432:5432 \
  -d ankane/pgvector:latest

# Wait for it to start
sleep 5

# Verify
docker logs langgraph_db
```

#### Option B: Manual PostgreSQL Installation
```bash
# Connect to your PostgreSQL
psql -U postgres -d agent_memory

# Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

# Verify
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3. Setup Environment Variables

Create `.env` file:
```env
# PostgreSQL
DATABASE_URL=postgresql://postgres:password@localhost:5432/agent_memory

# OpenAI (for embeddings)
OPENAI_API_KEY=sk-xxxx
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 4. Run Notebooks

```bash
jupyter notebook
```

Start with:
1. `stm_basic_inmemorysaver.ipynb` - Understand STM basics
2. `ltm_basic_memorystore.ipynb` - Understand LTM basics
3. `stm_persistence.ipynb` - Production STM
4. `ltm_persistence.ipynb` - Production LTM
5. `stm_ltm_combined_agent.ipynb` - Real agent example

---

## 📖 Learning Path

### For Beginners

1. **Understand Short-Term Memory**
   ```python
   from langgraph.checkpoint.memory import InMemorySaver
   checkpointer = InMemorySaver()
   # Messages stored in memory (lost on restart)
   ```

2. **Understand Long-Term Memory**
   ```python
   from langgraph.store.memory import InMemoryStore
   store = InMemoryStore()
   store.put(("user_123", "facts"), "key1", "value1")
   ```

3. **See Them in Action**
   - Open: `stm_basic_inmemorysaver.ipynb`
   - Open: `ltm_basic_memorystore.ipynb`

### For Production

1. **Persistent STM (PostgreSQL)**
   ```python
   from langgraph.checkpoint.postgres import PostgresSaver
   checkpointer = PostgresSaver(conn=connection_pool)
   checkpointer.setup()
   ```

2. **Persistent LTM (PostgreSQL + pgvector)**
   ```python
   from langgraph.store.postgres import PostgresStore
   store = PostgresStore(conn=connection_pool, index={...})
   store.setup()
   ```

3. **Combined Agent Example**
   - Open: `stm_ltm_combined_agent.ipynb`

---

## 🔑 Key Concepts

### Short-Term Memory (STM)

**What**: Conversation history within a thread
**How**: Checkpoint at every step
**Where**: PostgreSQL (production)
**Scope**: Single conversation
**Example**:
```python
config = {"configurable": {"thread_id": "conversation-1"}}

# STM stores:
# - User: "My name is Alice"
# - Bot: "Hi Alice!"
# - User: "What's my name?"
# - Bot: "Your name is Alice"  # Remember from STM!
```

### Long-Term Memory (LTM)

**What**: Facts and preferences across all conversations
**How**: Semantic search with embeddings
**Where**: PostgreSQL + pgvector (production)
**Scope**: All conversations with user
**Example**:
```python
namespace = ("user_123", "facts")

store.put(namespace, "pref_lang", "Python")  # Store fact
store.put(namespace, "job", "Engineer")      # Store fact

results = store.search(namespace, "What does user like?")
# Returns: [Python fact with score 0.92, Engineer fact with score 0.87]
```

### Thread vs User ID

```python
config = {
    "configurable": {
        "thread_id": "conversation-session-1",  # STM scope (unique per session)
        "user_id": "user_alice_123"              # LTM scope (per user)
    }
}

# Same user, different threads = Same LTM, Different STM
# Different users = Different LTM, Different STM
```

---

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│            User Input / Query                   │
└────────────────────┬────────────────────────────┘
                     │
            ┌────────▼─────────┐
            │ Load STM (Thread)│
            │ PostgreSQL       │
            └────────┬─────────┘
                     │
            ┌────────▼──────────────┐
            │ Load LTM (User)       │
            │ PostgreSQL + pgvector │
            │ Semantic Search       │
            └────────┬──────────────┘
                     │
        ┌────────────▼────────────────┐
        │  Agent Reasoning            │
        │  STM + LTM Context          │
        │  LLM Processing             │
        └────────────┬─────────────────┘
                     │
        ┌────────────▼─────────────────┐
        │  Update STM + LTM            │
        │  Save to PostgreSQL          │
        │  Index with pgvector         │
        └────────────┬─────────────────┘
                     │
            ┌────────▼──────────┐
            │  Response         │
            └───────────────────┘
```

## 📚 References

- **LangGraph Docs**: https://docs.langchain.com/oss/python/langgraph
- **Memory Guide**: https://docs.langchain.com/oss/python/langgraph/add-memory
- **PostgreSQL**: https://www.postgresql.org/docs/
- **pgvector**: https://github.com/pgvector/pgvector
- **Semantic Search**: https://openai.com/blog/new-embedding-models-and-api-updates

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Support

If you encounter issues:

1. Check the **Troubleshooting** section
2. Review notebook examples for your use case
3. Check PostgreSQL is running: `psql -U postgres -c "SELECT 1"`
4. Verify pgvector: `psql -d agent_memory -c "CREATE EXTENSION IF NOT EXISTS vector;"`
5. Enable debug logging to understand flow

---

## 🎓 Learning Resources

### For Beginners
- Start with: `stm_basic_inmemorysaver.ipynb`
- Then: `ltm_basic_memorystore.ipynb`
- Understand thread isolation and namespaces

### For Production
- Setup PostgreSQL (Docker recommended)
- Run: `stm_persistence.ipynb`
- Run: `ltm_persistence.ipynb`
- Deploy: `stm_ltm_combined_agent.ipynb`

### Advanced Topics
- Custom embedding models
- Multi-tenant applications
- Distributed checkpointing
- Fact extraction & deduplication

## 🚀 What's Next?

- [x] STM with PostgreSQL
- [x] LTM with semantic search
- [x] Combined agent example
- [ ] Multi-tenant support
- [ ] Fact deduplication
- [ ] Distributed deployment
- [ ] REST API wrapper

---

**Built with ❤️ using LangGraph, PostgreSQL, and pgvector**

⭐ If this project helped you, please star it on GitHub!
