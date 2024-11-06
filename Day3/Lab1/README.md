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
!pip install --quiet langchain==0.2.16 langchain-openai==0.1.23 langchain-community==0.2.16 qdrant-client==1.12.0 langchainhub==0.1.21 tavily-python==0.5.0 langgraph==0.2.39
```

4. Connecto to LansSmith for debuggin purposes:

```python
import os
from google.colab import userdata
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-workshops-day3"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = userdata.get('langsmith_key')
```

5. Go to https://tavily.com/ and create a free account there. Tavily is a search engine API, it integrates well with Langchain.

6. Grab the API key from Tavily and create new secret in Colab named **tavily_key**. Paste your API key there.

7. Finally, let's set-up keys and initiate the model:

```python
from google.colab import userdata
from langchain_openai import AzureChatOpenAI


os.environ["AZURE_OPENAI_ENDPOINT"] = "https://piotropenai.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "KEY"

gpt4 = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2023-06-01-preview"
)
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
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage


system_message = SystemMessage(
        content="You are a helpful assistant. Use all the tools to answer the question"
    )
```

4. Now let's create the agent. We are using here **Tool Calling** agent which requires a LLM with tools support. Most of new models support it (OpenAI, Anthropic, Gemini, Mistral etc.). We also create an instance of AgentExecutor class which is responsible for running the agent.

```python
agent_executor = create_react_agent(gpt4, tools, state_modifier=system_message)
```

5. Ok, let's invoke it:

```python
agent_executor.invoke({"messages": [("human", "Who is the coach of polish voleyball teams and what is his age?")]})

```

6. Try something more sophisticated:

```python
agent_executor.invoke({"messages": [("human", "Which teams played in the last final of man's voleyball world cup? What was the squad of each team? Give me also a nationality of each coach")]})
```

7. Or give some hints:

```python
agent_executor.invoke({"messages": [("human", "Which teams played in the last final of man's voleyball world cup? What was the squad of each team? Give me also a nationality of each coach. When responding first check the year of last world cup")]})
```

## Task 3: Add history to the agent

2.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

3. Let's create the agent again with memory now:

```python
agent_executor = create_react_agent(
        gpt4,
        tools,
        state_modifier=system_message,
        checkpointer=memory
    )

config = {
        "configurable": {
            "thread_id": "123"
        }
    }
```

4. And test it:
```python
agent_executor.invoke({"messages": [("human", "Which teams won the last final of man's voleyball world cup?")]}, config=config)
    

agent_executor.invoke({"messages": [("human", "Give me a squad of Polish team")]}, config=config)
    

agent_executor.invoke({"messages": [("human", "What was my last question?")]}, config=config)
```


## End lab