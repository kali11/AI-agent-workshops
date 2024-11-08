# Lab 1: Building AI agent
In this lab you will learn how to build an AI agent and how to configure it.

## Prerequisutes:
- OpenAI API key with a few Euros credits
- Google account

## Task 1: Set-up
1. Open Google Colab: https://colab.research.google.com/
2. Create new notebook, name it eg. **Workshop3 - la1**
3. First, we need to install dependencies. In the first cell type and run:

```python
!pip install --quiet langchain==0.2.16 langchain-openai==0.1.23 langchain-community==0.2.16 qdrant-client==1.12.0 langchainhub==0.1.21 tavily-python==0.5.0
```

4. Connecto to LansSmith for debuggin purposes:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-workshops-day3"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<LANGSMITH_KEY>"
```

5. Go to https://tavily.com/ and create a free account there. Tavily is a search engine API, it integrates well with Langchain.

6. Grab the API key from Tavily and create new secret in Colab named **tavily_key**. Paste your API key there.

7. Finally, let's set-up keys and initiate the model:

```python
from google.colab import userdata
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = userdata.get('openai_key')
gpt4 = ChatOpenAI(model = "gpt-4o", temperature = 0)
```

## Task 2: Create the agent
1. First, create a Tavily tool and a simple list of tools with just one element.

```python
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ["TAVILY_API_KEY"] = userdata.get('tavily_key')

tavily_tool = TavilySearchResults()
tools = [tavily_tool]
```

2. You can invoke the tool and see how it works:

```python
tavily_tool.invoke("funny cats")
```

3. Let's add agent's prompt. Note the {agent_scratchpad} placeholder for agent's notes

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use all the tools to answer the question"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
```

4. Now let's create the agent. We are using here **Tool Calling** agent which requires a LLM with tools support. Most of new models support it (OpenAI, Anthropic, Gemini, Mistral etc.). We also create an instance of AgentExecutor class which is responsible for running the agent.

```python
agent = create_tool_calling_agent(gpt4, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

5. Ok, let's invoke it:

```python
agent_executor.invoke({"input": "Who is the coach of polish voleyball teams and what is his age?"})
```

6. Try something more sophisticated:

```python
agent_executor.invoke({"input": "Which teams played in the last final of man's voleyball world cup? What was the squad of each team? Give me also a nationality of each coach"})
```

7. Or give some hints:

```python
agent_executor.invoke({"input": "Which teams played in the last final of man's voleyball world cup? What was the squad of each team? Give me also a nationality of each coach. When responding first check the year of last world cup "})
```

## Task 3: Add history to the agent
1. Let's modify the prompt:

```python
history_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant. Use all the tools to answer the question""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
    ]
)
```

2. And a memory. We will use simple in-memory **ChatMessageHistory** class. Note, that "chat_history" key needs to match the placeholder name in a prompt.

```python
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory

chat_history = ChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=chat_history,
    memory_key='chat_history',
    return_messages=True,
    input_key='input',
    output_key='output'
)
```

3. Let's create the agent again with memory now:

```python
agent = create_tool_calling_agent(gpt4, tools, history_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
```

4. And test it:
```python
agent_executor.invoke({"input": "Which teams won the last final of man's voleyball world cup? "})

agent_executor.invoke({"input": "Give me a squad of Polish team"})

agent_executor.invoke({"input": "What was my last question?"})
```

5. We can also add flag that will show us how agent was "thinking":

```python
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, return_intermediate_steps=True)
agent_executor.invoke({"input": "Search for information about wombats"})
```

## Task 4: Use ReAct agent
1. Now let's use ReAct agent that doesn't use tool calling but [ReAct prompting](https://www.promptingguide.ai/techniques/react) instead. 

2. We will use Langchain hub here. It is a public repository with prompts. Go to: https://smith.langchain.com/hub and look for "hwchase17/react". Inspect how this prompt is build.

3. Let's use this prompt in a code:

```python
from langchain import hub
from langchain.agents import create_react_agent

prompt = hub.pull("hwchase17/react")
```

4. And let's create the agent and agent executor:

```python
agent = create_react_agent(gpt4, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True)
```

5. See the agent in action:

```python
agent_executor.invoke({"input": "Which teams played in the last final of man's voleyball world cup? What was the squad of each team? Give me also a nationality of each coach. When responding first check the year of last world cup "})
```

## End lab