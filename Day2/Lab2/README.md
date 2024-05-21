# Lab 2: Building RAG
In this lab we will build a RAG that works with our knowledge base.

## Prerequisutes:
- OpenAI API key with a few Euros credits
- Google account
- Lab1 finished

## Task 1: Set-up
1. Open Google Colab: https://colab.research.google.com/
2. Create new notebook, name it eg. **Workshop2 - la2**
3. First, we need to install dependencies. In the first cell type and run:

```python
!pip install --quiet langchain==0.1.20 langchain-openai==0.1.6 qdrant-client==1.9.1
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

## Task 2: Create RAG chain
1. First, we need to create a retreiver. It is a Langchain object that allows to retrieve content from data sources:

```python
retriever = qdrant.as_retriever()
```
Alternatively, you can also use MMR:

```python
retriever = qdrant.as_retriever(search_type="mmr")
```

2. Now let's create a prompt and a simple function that concatenates multiple documents into one:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "czy mogę jeszcze jakoś pomóc?" at the end of the answer.

{context}

Question: {question}

Helpful Answer:""")

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)
```

3. Let's construct the RAG chain:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rag = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | gpt4
    | StrOutputParser()
)
```

4. And invoke:

```python
rag.invoke("Ile przysługuje dni urlopu wypoczynkowego?")
```

5. Try to ask different questions :)

## Task 3: Add history to the chat
In order to create chatbot with history we can use  prebuild chains.

1. First, let's create a **history_aware_retriever**. This retriever summarizes the history and reformulates the question if needed:

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    gpt4, retriever, contextualize_prompt
)
```

2. Then, we need to create a **stuff_document_chain** that can pass a list of documents to the model:

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(gpt4, qa_prompt)
```

3. Now, let's create **retrieval_chain**. It takes as the arguments the retriever and stuff_documents chain

```python
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

4. Now, it's time to create history storage. For this example we will use simple **ChatMessageHistory** that just stores chat history in memory. 
>Langchain supports many different message history providers. You can check them [here](https://python.langchain.com/v0.1/docs/integrations/memory/)

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```
It simply just creates a dict object with sessions and corresponding history objects

5. So, let's create a final chain:

```python
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```

6. And run it:

```python
conversational_rag_chain.invoke(
    {"input": "Ile przysługuje dni urlopu wypoczynkowego?"},
    config={
        "configurable": {"session_id": "123"}
    },  # constructs a key "123" in `store`.
)
```

7. Try different questions, for example:
- O co pytałem poprzednio?
- Jak aplikować o urlop maciezyński?

Try also changing the session id.

8. Check the memory content:
```python
display(store)
```

9. Investigate the Langsmith monitoring.

## End lab