# Lab 2: Building multi-tools agent
In this lab you will learn how to build an AI agent with multiple tools.

## Prerequisutes:
- OpenAI API key with a few Euros credits
- Google account
- Tavily API key

## Task 1: Set-up
1. Open Google Colab: https://colab.research.google.com/
2. Create new notebook, name it eg. **Workshop3 - la1**
3. First, we need to install dependencies. In the first cell type and run:

```python
!pip install --quiet \
langchain==0.2.1 \
langchain-openai==0.1.7 \
qdrant-client==1.9.1 \
langchainhub==0.1.15 \
langchain-community==0.2.1 \
tavily-python==0.3.3 \
langchain-experimental==0.0.59 \
SQLAlchemy==2.0.30
```

4. Connecto to LansSmith for debuggin purposes:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-workshops-day3"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<LANGSMITH_KEY>"
```

5. Finally, let's set-up keys and initiate the model. This time we will use gpt-4-turbo:

```python
from google.colab import userdata
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = userdata.get('openai_key')
gpt4 = ChatOpenAI(model = "gpt-4-turbo", temperature = 0)
```

## Task 2: Set-up SQL database
We will use data from SQL database with SQLite engine. 

1. Download **Chinook.db** file from repository and upload it into files on Colab.
2. Activate SQL extension in Colab. In a new cell paste:
```sql
%load_ext sql
```
3. Load the database:
```sql
%%sql
sqlite:///Chinook.db
```
4. Now you should be able to list all tables in database:
```sql
%%sql
SELECT name FROM sqlite_master
    WHERE type='table';
```
5. And execute simple query:
```sql
%%sql
select BillingCountry, sum(Total) from Invoice group by BillingCountry;
```

## Task 3: Set-up all tools

1. Set-up Tavily tool:
```python
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ["TAVILY_API_KEY"] = userdata.get('tavily_key')

tavily_tool = TavilySearchResults()
```

2. **You can omit this if you haven't completed Lab2 from Day2**. Here we will configure a retriever tool. It will retrieve data from our vector database.
>Notice that create_retriever_tool() method needs a tool name and description. It is important because LLM bases on these descriptions.

```python
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.tools.retriever import create_retriever_tool

collection_name = "labor_law"
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

qdrant_client = QdrantClient(
    url="<QDRANT_URL>",
    api_key=userdata.get('qdrant_key')
)
qdrant = Qdrant(qdrant_client, collection_name, embeddings)

retriever_tool = create_retriever_tool(
    qdrant.as_retriever(),
    "search_documents",
    "Searches and returns documents about labor law in Poland",
)
```

3. Remember to put your **Qdrant URL** above and **qdrant_key** into secrets.

4. Now, let's create a PythonREPL tool that can write and execute Python code locally:

```python
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
```

5. Create SQLDatabase toolkit. Toolkit is a set of tools. Here we load all tools that can interact with a database:

```python
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

toolkit = SQLDatabaseToolkit(db=db, llm=gpt4)
context = toolkit.get_context()
db_tools = toolkit.get_tools()
```

6. Inspect what tools are inside:

```python
db_tools
```

7. Now, let's combine all tools into signle list:

```python
all_tools = []
all_tools += db_tools
all_tools.append(retriever_tool)
all_tools.append(tavily_tool)
all_tools.append(repl_tool)

all_tools
```

## Task 4: Multi-tools agent
1. Create the prompt and the agent:

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
prompt = prompt.partial(**context)

agent = create_tool_calling_agent(gpt4, all_tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)
```

2. Test it:
```python
agent_executor.invoke(
    {
        "input": "What is the 10th fibonacci number?"
    }
)
```
```python
agent_executor.invoke(
    {
        "input": "Give me list of invoice sales amount per each country"
    }
)
```
```python
agent_executor.invoke(
    {
        "input": "Give me list of invoice sales amount per each country. Then draw a barchart"
    }
)
```
```python
agent_executor.invoke(
    {
        "input": """Give me list of invoice sales amount per each country. Then draw a barchart. Then take country with
        highest value and search the web for the amount of days of holiday leave in this country.
        """
    }
)
```
```python
agent_executor.invoke(
    {
        "input": """Give me list of invoice sales amount per each country. Then draw a barchart. Then take country with
        highest value and search the web for the amount PTO in this country.
        Finally, compare the PTO with the holiday leave days in Poland.
        """
    }
)
```

3. You can also try to insert some data:

```python
agent_executor.invoke(
    {
        "input": "Insert new Artist named Zenek Martyniuk"
    }
)
```