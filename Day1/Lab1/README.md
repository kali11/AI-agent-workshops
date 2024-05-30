# Lab 1: Working with Langchain
In this lab you will learn:
- Basics of the Langchain framework
- How to create chains
- How to use LCEL (Langchain expression language) to build different flows of LLM invocations.
- How to use GPT tools (functions)
- How to monitor you app
- How to work with different models (gpt3.5, gpt4, dalle)

## Prerequisutes:
- OpenAI API key with a few Euros credits
- Google account

## Task 1: First LLM invocations
1. Open Google Colab: https://colab.research.google.com/
1. Create new notebook, name it eg. **Workshop1 - lab1**
1. First, we need to install dependencies. In the first cell type and run:\
```!pip install --quiet langchain==0.1.20 langchain-openai==0.1.6```\
Here we install Langchain framework and langchain-openai responsible for OpenAI integration.
1. In the next cell paste your OpenAI API key and create an instance of gpt-3.5 model:

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"

gpt35 = ChatOpenAI(model = "gpt-3.5-turbo")
```
**ChatOpenAI** is a Langchain class. You can add here more parameters like temperature, max_tokens etc. [See in docs](https://api.python.langchain.com/en/latest/chat_models/langchain_community.chat_models.openai.ChatOpenAI.html)
5. Invoke the model with simple prompt:
```python
gpt35.invoke("tell ma a joke about pizza")
```
6. Observe the output. Notice tokens usage.

## Task 2: Switch on LangSmith.

LangSmith is an online tool that is used to monitor Langchain applications. It will show you what are exact invocations of the LLM model. Very useful tool.

1. Go to: https://smith.langchain.com/
2. Sign-up in the service
3. On the left-hand site click **Settings**
4. Generate new API key (personal) and copy it
5. Create new cell in the notebook and paste LangSmith configuration:
```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-agent-workshops"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_KEY"
```
6. Run one more time the call above that invokes gpt-3.5 model.
7. In LangSmith click **Projects** in left menu, find "ai-agent-workshops" project and observe the tracing.
8. Now LangSmith is switched on. It will be useful in next tasks. Free plan is sufficient.

## Task 3: Playing with chains
1. In a new cell create a prompt template. Prompt templates are useful in order to build a general prompt and modify just a part of it. Here we create a simple **system** and **user** messages with two place holders: **role** and **prompt**:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that acts as a {role}. Answer only questions about your role"),
    ("user", "{prompt}")
])
```
2. Most of Langchain classes implements **Runnable** interface so you can run **invoke()** method on it. Let's invoke the prompt and see how placeholders are filled in:
```python
prompt.invoke({"prompt": "How to make pizza?", "role": "cook"})
```
2. Notice that our prompt contains two messages: **SystemMessage** and **HumanMessage**.
2. Now let's create a first chain using LCEL (Langchain expression language). We will use pipe operator (|):
```python
chain = prompt | gpt35
chain.invoke({"prompt": "How to make pizza?", "role": "cook"})
```
2. As a result we get an **AIMessage**.
2. GPT-3.5 is a text-2-text model. Very often we want to extract raw text. Langchain comes here with different parsers. Here we will use simple **StrOutpurParser**:

```python
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = prompt | gpt35 | output_parser
chain.invoke({"prompt": "How to make pizza?", "role": "cook"})
```
7. You can also check other available parsers here: https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/

8. Try also to invoke the model with non-matching values, for example: `{"prompt": "How to make pizza?", "role": "doctor"}`. :)

9. Langchain supports streaming. Just change **invoke()** method to **stream()**:
```python
for s in chain.stream({"prompt": "How to make pizza?", "role": "cook"}):
    print(s, end="", flush=True)
```
10. It is important that each connected **Runnables** (prompt | gpt35 and gpt35 | output_parser) needs to be compatible in terms of schema. You can always check the schema of each chain component:
```python
print(chain.input_schema.schema())
print(chain.output_schema.schema())
```

## Task 4: Longer chains
1. Now let's build something a little bit longer and let's switch to GPT-4. Make an instance of a model:
```python
gpt4 = ChatOpenAI(model = "gpt-4")
```
1. Now we will build a chain that will:
- Ask for a dish recipe
- Based on the recipe it will list all needed ingredients
3. In a new cell, let's create the first chain:

```python
recipe_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a cook that gives dish recipes."),
    ("user", "Give me a recipe for {dish}")
])

chain = recipe_prompt | gpt4 | output_parser
print(chain.invoke({"dish": "pizza"}))
```
4. In a new cell, create second prompt:

```python
ingredients_prompt = ChatPromptTemplate.from_template("Based on the recipe: {recipe} list me all ingredients needed for the recipe. Return just the list and nothing else")
```

5. Now create a full chain. Notice that we are using two LLMs here. We are also using simple lambda function to pass output as a new input:

```python
chain2 = (
    recipe_prompt
    | gpt4
    | output_parser
    | (lambda input: {"recipe": input})
    | ingredients_prompt
    | gpt35
    | output_parser
)
print(chain2.invoke({"dish": "pizza"}))
```
6. We can also embed one chain in another: (same effect as above)

```python
chain3 = (
    chain
    | (lambda input: {"recipe": input})
    | ingredients_prompt
    | gpt35
    | output_parser
)
print(chain3.invoke({"dish": "pizza"}))
```
7. Investigate LangSmith debug

## Task 5: Tool calling
Most of new models support tools calling (previously function calling). This mechanism is used to return structured output from LLM. Remember that this mechanism does not call any function/tool. It just prepares the input.

1. In Langchain we can define tools in several ways. For example using @tool decorator or using Pydantic types. Here we define two tools (**add** and **multiply**) in two different ways:

```python
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# add tool
@tool
def add(a: int, b: int) -> int:
    return a + b

# multiply tool
class multiply(BaseModel):
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

tools = [add, multiply]
```
2. Now we need to bind these tools to our LLM so it can invoke them. Note, that by default LLM will decide if it wants to use a tool or not. We can also enforce it.

```python
gpt35_with_tools = gpt35.bind_tools(tools)
```

3. Let's call it and observe how LLM model has used the tools:
```python
result = gpt35_with_tools.invoke("What is 3 * 8? Also, what is 8 + 63?")
result
```

4. In a new cell inspect tool calls:
```python
result.tool_calls
```

5. We can also explicitly invoke a tool function with arguments. Let's extract them from the result and invoke the **add** tool:

```python
def extract_args(functions, name):
    for func in functions:
        if func['name'] == name:
            return func['args']
    return None

add_args = extract_args(result.tool_calls, 'add')
add.invoke(add_args)
```

6. Tool calling are also traced by LangSmith.

## Task 6: Image generation with Dall-e
In Lanchain we have a **DallEAPIWrapper** class for using Dall-e model. In the below example we will first use gpt-3.5 to generate a prompt for Dall-e and then generate an image. Note that Dall-e accepts prompt of maximum 1000 characters. That is why we will limit max_token to 200.

1. In a new cell type:

```python
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

gpt35_mt = ChatOpenAI(max_tokens=200)

prompt = ChatPromptTemplate.from_template("Generate a detailed prompt to generate an image based on the following description: {image_desc}.")
dalle_chain = prompt | gpt35_mt | output_parser
dalle_prompt = dalle_chain.invoke({"image_desc": "cat at home"})
print(dalle_prompt)
```

2. Now let's define a function that displays image in a notebook:

```python
def display_image(image_url):
  try:
      import google.colab

      IN_COLAB = True
  except ImportError:
      IN_COLAB = False

  if IN_COLAB:
      from google.colab.patches import cv2_imshow  # for image display
      from skimage import io

      image = io.imread(image_url)
      cv2_imshow(image)
  else:
      import cv2
      from skimage import io

      image = io.imread(image_url)
      cv2.imshow("image", image)
      cv2.waitKey(0)  # wait for a keyboard input
      cv2.destroyAllWindows()
```

3. And finally let's call dall-e and display the image:

```python
image_url = DallEAPIWrapper().run(dalle_prompt)
print(image_url)
display_image(image_url)
```
image_url variable will point to a public image on Azure blob storage.

## End lab
