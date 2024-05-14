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
!pip install --quiet langchain==0.1.20 langchain-openai==0.1.6
```

Here we install Langchain framework and langchain-openai responsible for OpenAI integration.
1. In the next cell paste your OpenAI API key and create an instance of gpt-3.5 model:

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"

gpt35 = ChatOpenAI(model = "gpt-3.5-turbo")
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

gpt35.invoke(
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
gpt35.invoke([HumanMessage(content="What I was asking about?")])
```

3. We need to supply whole history:

```python
from langchain_core.messages import AIMessage

gpt35.invoke(
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

chain = prompt | gpt35
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

3. Now the memory will be handled by Langchain. **RunnableWithMessageHistory** is a class that allows us to use memory with Runnables. A few remarks here:
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

chain = prompt | gpt35

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
Langchain comes with pre-defined chains. They are ready to use, with default prompting already defined. All of them can be found here: [chains](https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.openai.ChatOpenAI.html)

1. Let's use the ConversationChain. It is simple chain that already has a history.

```python
from langchain.chains import ConversationChain

conversation = ConversationChain(
    llm=gpt35,
    verbose=True
)
```

2. Let's test:

```python
conversation.invoke(input="What is the capital of Poland?")
conversation.invoke(input="What was my previous question?")
```

3. Langchain also comes with different memory types. All of them can be found [here](https://python.langchain.com/v0.1/docs/modules/memory/)

4. Let's use **ConversationBufferWindowMemory** that stores only k newest messages.

```python
from langchain.memory import ConversationBufferWindowMemory

conversation = ConversationChain(
    llm=gpt35,
    verbose=True,
    memory = ConversationBufferWindowMemory(k=2)
)
```

5. And test:

```python
conversation.invoke(input="What is the capital of Poland?")
conversation.invoke(input="How large is the city?")
conversation.invoke(input="How is the population?")
conversation.invoke(input="What was my first question?")
```

6. Let's also see **ConversationSummaryMemory**. It summarizes the whole conversation using the LLM. It is useful when we want to prevent the context from being too long. Notice, that we need to pass llm that will create the summary. See also what prompt is used:

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=gpt35)
print(memory.prompt)

conversation_with_summary = ConversationChain(
    llm=gpt35,
    memory=ConversationSummaryMemory(llm=gpt35),
    verbose=True
)
```
7. Test and see the history summary:

```python
conversation.invoke(input="What is the capital of Poland?")
conversation.invoke(input="How large is the city?")
conversation.invoke(input="What is the population?")
conversation.invoke(input="What was my first question?")

conversation.memory.load_memory_variables({})
```
