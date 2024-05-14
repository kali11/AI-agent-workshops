# Lab 2: LinkedIn post generator
In this lab you will build a simple LinkedIn post generator.

## Prerequisutes:
- OpenAI API key with a few Euros credits
- Google account

## Task 1: Set-up
1. Open Google Colab: https://colab.research.google.com/
1. Create new notebook, name it eg. **Workshop1 - lab2**
1. First, we need to install dependencies. In the first cell type and run:

```python
!pip install --quiet langchain==0.1.20 langchain-openai==0.1.6
```

Here we install Langchain framework and langchain-openai responsible for OpenAI integration.
1. In the next cell paste your OpenAI API key and create an instance of gpt-4 model:

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field

os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"

gpt4 = ChatOpenAI(model = "gpt-4")
```

2. Switch on LangSmith:

```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ai-agent-workshops"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_KEY"
```

## Task 2: LinkedIn post generator!
Now let's combine all the knowledge and let's build a simple LinkedIn post generator. :)
Here is the idea:
- first, generate post content
- then, do text correction, remove errors, add emojis
- generate also image for the post and hashtags
So the chain will looks like this: generate post content -> correct the post -> generate image -> generate hashtags

1. First, generating the post content:

```python
output_parser = StrOutputParser()

generate_content_prompt = ChatPromptTemplate.from_template("Based the outline: {outline} generate me a LinkedIn post. Use AIDA model. Do not generate hashtags. Use {language} language")

generate_content_chain = generate_content_prompt | gpt4 | output_parser
```

2. Try how it works:

```python
outline = """
- I've attended the AI workshop
- I've created my first app using Langchain
- I've built my own chatbot
- I've created a LinkedIn post generator
- I recommend this workshops
mark @Piotr Kalinowski
"""
post = generate_content_chain.invoke({"outline": outline, "language": "polish"})
print(post)
```

3. Now the correction prompt and dalle metaprompt:

```python
correction_prompt = ChatPromptTemplate.from_template("Correct this LinkedIn post: {post}. Remove errors and unnecessary interjections. Add emojis. Do not change language")

dalle_metaprompt = ChatPromptTemplate.from_template("Generate a detailed prompt to generate a photograph for the LinkedIn post. Here is the post text: '{post}'.")
```

4. In order to generate image we need to execute **DallEAPIWrapper().run()** method. However **DallEAPIWrapper** does not implements **Runnable** interface. That is why we need to wrap it in a function and use **RunnableLambda** that transform a function into **Runnable**:

```python
def run_dalle(prompt):
  return DallEAPIWrapper().run(prompt + " High resolution, 2K, 4K, 8K, clear, good lighting, detailed, extremely detailed, sharp focus, intricate, beautiful, realistic+++, complementary colors, high quality, hyper detailed, masterpiece, best quality, artstation, stunning")
```
(Notice: We are adding to the prompt some image prompt engineering)

5. In order to generate hastags list we will use a tool:

```python
class hashtag_generator(BaseModel):
    """Generates hashtags for a LinkedIn post."""

    hastags: list[str] = Field(..., description="A list of LinkedIn hashtags. Focus mostly on AI")

tools = [hashtag_generator]
```
6. Ok, let's combine everything together:

```python
chain = (
  generate_content_chain
  | (lambda input: {"post": input})
  | correction_prompt
  | gpt4
  | output_parser
  | gpt4.bind_tools(tools)
)
chain.invoke({"outline": outline, "language": "polish"})
```
It works, however it is not what we are looking for. As a result we want to have a post content, list of hashtags and an image.

7. Let's use a **RunnableParallel** class. It can execute multiple chains in parallel. First let's define two chains. One for dalle and second for hashtags:

```python
dalle_chain = dalle_metaprompt | gpt4 | output_parser | RunnableLambda(run_dalle)

def hashtags_to_string(hashtag_generator):
  result = ""
  for tag in hashtag_generator[0].hastags:
    result += "#" + tag + " "
  return result
hashtags_chain = gpt4.bind_tools(tools) | PydanticToolsParser(tools=tools) | RunnableLambda(hashtags_to_string)
```
For the hashtags we used PydanticToolsParser that simple parses the tools output. If facilities usage of **hashtags_to_string** function

8. And finally! Here is our last chain:

```python
chain = (
    generate_content_chain
    | (lambda input: {"post": input})
    | correction_prompt
    | gpt4
    | output_parser
    | RunnableParallel(post = RunnablePassthrough() ,hashtags = hashtags_chain, image = dalle_chain)
)
```
We also used here **RunnablePassthrough** class that just passes the input into the output unchanged. This typically is used in conjuction with RunnableParallel to pass data through to a new key in the map.
9. Create helper function:

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

10. Invoke the generator!

```python
result = chain.invoke({"outline": outline, "language": "polish"})
print(result["post"])
print(result["hashtags"])
display_image(result["image"])
```

11. Investigate the LangSmith!
12. Put the post on your LinkedIn ;)

## End lab

### Final questions and concerns:
- Why generated images are so bad?
- What happend when you enter the same dalle prompt into chatGPT (you can extract the prompt from Langsmith)
- how much money this lab consumed :)