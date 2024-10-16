# Lab 3: Chatbot with memory
In this lab you will build a chatbot that remembers conversation history. A chatbot is basically a chain but with memory.

## Prerequisutes:
- OpenAI API key with a few Euros credits
- Google account

## Task 1: Set-up
1. Open Google Colab: https://colab.research.google.com/
1. Create new notebook, name it eg. **Workshop1 - lab3**
1. First, we need to install dependencies. In the first cell type and run:

```python
!pip install --quiet langchain==0.2.16 langchain-openai==0.1.23 langchain-community==0.2.16
```

Here we install Langchain framework and langchain-openai responsible for OpenAI integration.
1. In the next cell create an instance of gpt-4o model:

```python
import os
from langchain_openai import ChatOpenAI
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get('openai_key')

gpt4 = ChatOpenAI(model = "gpt-4o-mini")
```

2. Switch on LangSmith:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-agent-workshops"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_KEY"
```

## Task 2: Build a simple chatbot
LLM models and thier's APIs are usually stateless. That means we need to always send to the model whole message history. Let's see an example:
1. Create new cell and paste:

```python
from langchain_core.messages import HumanMessage

gpt4.invoke(
    [
        HumanMessage(
            content="What is the capital of Poland?"
        )
    ]
)
```

You should see the result in the output as an instance of AIMessage.

2. But we don't have a history:

```python
gpt4.invoke([HumanMessage(content="What I was asking about?")])
```

3. We need to supply whole history:

```python
from langchain_core.messages import AIMessage

gpt4.invoke(
    [
        HumanMessage(
            content="What is the capital of Poland?"
        ),
        AIMessage(content="The capital of Poland is Warsaw"),
        HumanMessage(content="What I was asking about?")
    ]
)
``` 

4. We can also make it more effective using **MessagesPlaceholder**. This class is very useful when constructing prompts:
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | gpt4
```

## Task 3: Add memory
Langchain comes with many different memory types. Memory can be:
- handled by ourselves - then we need to build it manually
- handled by Langchain

1. First, let's do it manually. In a new cell type:

```python
from langchain.memory import ChatMessageHistory

chat_history = ChatMessageHistory()
chat_history.add_user_message("What is the capital of Poland?")
chat_history.add_ai_message("The capital of Poland is Warsaw")
chat_history.add_user_message("How large is this city?")
chat_history.messages
```

2. And let's invoke:

```python
chain.invoke({"messages": chat_history.messages})
```

3. Now, the memory will be handled by Langchain. **RunnableWithMessageHistory** is a class that allows us to use memory with Runnables. A few remarks here:
- **RunnableWithMessageHistory** encloses a chain
- second argument is a function that returns chat history
- third argument is a key for input. <u>It needs to match the placeholder in a promp template</u>
- fourth argument is a key for history. <u>Again, it needs to match the placeholder in a promp template</u>

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

chat_history = ChatMessageHistory()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | gpt4

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
```

4. Now let's invoke our chain. We are adding a "configurable" key to pass session_id. It is unused, but it is required.

```python
chain_with_history.invoke(
    {"input": "What is the capital of Poland"},
    {"configurable": {"session_id": "unused"}},
)
```

5. And one more time:

```python
chain_with_history.invoke(
    {"input": "How large is this town?"},
    {"configurable": {"session_id": "unused"}},
)
```

6. Observe the messages history:

```python
chat_history.messages
```

## Task 4: Use pre-defined Langchain chains.
Langchain comes with pre-defined chains. They are ready to use, with default prompting already defined. **Some of them already are or will be depricated!** All of them can be found here: [chains](https://python.langchain.com/v0.1/docs/modules/chains/)

> Langchain rather withdraws from those chains in favor of LCEL chains and Langgraph (Langchain's extension). You can learn more about Langgrpah in Day 4 of this workshops.

1. Let's use the LLMMathChain. It is simple chain optimized for mathematical operations.

```python
from langchain.chains import LLMMathChain

llm_math = LLMMathChain.from_llm(gpt4, verbose=True)
```

2. Let's test:

```python
llm_math.invoke(input="What is 9 raised to .9876 ower?")
```

3. Check Langgraph and see how to chain works under the hood.

## END LAB
