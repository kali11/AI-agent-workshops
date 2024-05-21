# Lab 3: RAG Agent
In this lab we will build an AI agent with RAG capabilities. This lab creates a simple agent, more sophisticated agents are build during Day 3.

## Prerequisutes:
- OpenAI API key with a few Euros credits
- Google account
- Lab1 finished

## Task 1: Set-up
1. Open Google Colab: https://colab.research.google.com/
2. Create new notebook, name it eg. **Workshop2 - la3**
3. First, we need to install dependencies. In the first cell type and run:

```python
!pip install --quiet langchain==0.1.20 langchain-openai==0.1.6 qdrant-client==1.9.1 langchainhub==0.1.15
```

4. Connecto to LansSmith for debuggin purposes:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-workshops-day2"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<LANGSMITH_KEY>"
```

5. Configure Qdrant client, embeddings model and LLM:

```python
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from google.colab import userdata
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = userdata.get('openai_key')

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
collection_name = "labor_law"

qdrant_client = QdrantClient(
    url="<QDRANT_URL>",
    api_key=userdata.get('qdrant_key')
)
qdrant = Qdrant(qdrant_client, collection_name, embeddings)

gpt4 = ChatOpenAI(model = "gpt-4o")
```

## Task 2: Create the agent
1. Create retriever:
```python
retriever = qdrant.as_retriever()
```

2. Create a retriever tool for the agent:
```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "search_documents",
    "Searches and returns documents about labor law in Poland",
)
tools = [retriever_tool]
```

3. Create a prompt. Here we will use [langchain hub](https://smith.langchain.com/hub), which is a place where you can find different, publicly available prompts. Go to hub and see how **hwchase17/openai-tools-agent** looks like.

```python
from langchain import hub

prompt = hub.pull("hwchase17/openai-tools-agent")
```

4. Now, let's create the agent with tools and the agent executor:
```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

agent = create_tool_calling_agent(gpt4, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

5. Invoke the agent and observe the output:

```python
agent_executor.invoke(
    {
        "input": "Ile przysługuje dni urlopu wypoczynkowego?"
    }
)
```

## End lab