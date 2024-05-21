# Lab 1: Data embedding and vectorization
In this lab we will create a knowledge base for a labor law data. You will learn:
- Basics of embedding
- How to load documents into vector database
- How to search for documents in vector database

## Prerequisutes:
- OpenAI API key with a few Euros credits
- Google account

## Task 1: Create vector database
For the vector database we will use Qdrant in a free cloud-based version.
1. Go to https://cloud.qdrant.io/ and create a new account.
2. When logged in, click **Clusters** on the left menu and create a new FREE cluster
3. Grab tha API key and cluster url
4. When the cluster is created, select it and click on **Open Dashboard** button.


## Task 2: Set-up
1. Open Google Colab: https://colab.research.google.com/
2. Create new notebook, name it eg. **Workshop2 - lab1**
3. First, we need to install dependencies. In the first cell type and run:

```python
!pip install --quiet langchain==0.1.20 langchain-openai==0.1.6 qdrant-client==1.9.1 unstructured[pdf]
```
It will take some time because **unstructured** is a large library.

Here we install:
- Langchain framework 
- langchain-openai responsible for OpenAI integration.
- qdrant-client responsible for Qdrant integration.
- unstructured[pdf] responsible for PDF parsing.

4. In the left menu of Google Colab click **KEY** button. It is a place when you can store your secrets.
5. Create two new secrets:
- openai_key
- qdrant_key

Make both of them accessible from notebooks.

5. In the next cell paste your OpenAI API key and create an instance of the **text-embedding-3-large** model:

```python
import os
from langchain_openai import OpenAIEmbeddings
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get('openai_key')

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
```

6. In this lab we will index documents about labor law in Poland. Download these two files:
- asf
- asf

In **Google Colab** click **Files** button on the left menu and create **docs** directory. Upload both files there.

## Task 3: Load documents into vector database
1. First we need to load our documents from file system. We will use **DirectoryLoader** for that. 

>Langchain supports loading data from many different sources (eg. Google Drive, Sharepoint, Web, Wikipedia and much, much more). You can check all options in [the docs](https://python.langchain.com/v0.1/docs/integrations/document_loaders/).

> DirectoryLoader uses **Unstructured** by default to load documents from a PDF file. This lib is flexible, in can also load eg. docx, pptx, but it is also large. You can change the loader to load class if needed.

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader('./docs', glob="**/*.pdf", show_progress=True)
documents = loader.load()
```

2. Once you've loaded documents, you'll often want to transform them to better suit your application. Langchain comes with a lot of Document Transformers that can help here. 

> Very often we want to split our documents into smaller chunks - pdf files can be large. The recommended class is a **RecursiveCharacterTextSplitter**. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is **["\n\n", "\n", " ", ""]**. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible.

3. Let's split our documents into smaller chunks. **chunk_overlap** parameter usually improves RAG capabilities.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(len(docs))
```

3. Ok, now let's create qdrant client:

```python
from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="<QDRANT_URL>",
    api_key=userdata.get('qdrant_key')
)
```

4. Then, we need to create a collection. Collection is a little bit like a table in a SQL database. We need to specify the size of the vector - 3072. This size is determined by the embedding model we use (**text-embedding-3-large** in our case).

```python
from qdrant_client.models import Distance, VectorParams

qdrant_client.recreate_collection(
    collection_name="labor_law",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)
```

5. Now, create langchain's vectorstore for Qdrant. Note, that we need to specify the embeddings.

```python
from langchain_community.vectorstores import Qdrant

qdrant = Qdrant(qdrant_client, "labor_law", embeddings)
```

6. Let's load our documents!

```python
qdrant.add_documents(docs)
```

7. All code in this Task could be also performed in on command:

```python
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url=<QDRANT_URL>,
    prefer_grpc=True,
    api_key=userdata.get('qdrant_key'),
    collection_name="labor_law2",
    force_recreate=True
)
```

8. Go to Qdrant UI dashboard, select your collection and check if documents are loaded.

## Task 4: Query and search vector database
1. Let's query the database with simple similarity search that return top k most similar documents:

```python
query = "Ile przysługuje dni urlopu wypoczynkowego?"
found_docs = qdrant.similarity_search(query, k=3)
print(found_docs[0].page_content)
```
>You can also ask the question in english

2. You can display the source of each document:
```python
print(found_docs[0].metadata["source"])
```

3. And you can also display a similarity score:
```python
query = "Ile przysługuje dni urlopu wypoczynkowego?"
found_docs = qdrant.similarity_search_with_score(query, k=3)
for d, s in found_docs:
  print(d.page_content)
  print(f"\nScore: {s}")
  print(d.metadata["source"])
  print("---------------------------------------------------------------")
```

4. Another useful algorithm for searching is MMR (maximal marginal relevance). 
>MMR tries to look for documents most similar to the inputs, while also optimizing for diversity. It does this by finding the examples with the embeddings that have the greatest cosine similarity with the inputs, and then iteratively adding them while penalizing them for closeness to already selected examples.
```python
query = "Ile przysługuje dni urlopu wypoczynkowego?"
found_docs = qdrant.max_marginal_relevance_search(query, k=3, fetch_k=10)
for d in found_docs:
  print(d.page_content)
  print(d.metadata["source"])
  print("---------------------------------------------------------------")

```
Compare the results with previous ones.

5. Qdrant also allows us to add additional filters. For example we can search only in subset od documents:

```python
from qdrant_client import models

my_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="metadata.source",
            match=models.MatchValue(value="docs/praca_rodzicielstwo.pdf"),
        )
    ]
)

query = "Ile przysługuje dni urlopu wypoczynkowego?"

found_docs = qdrant.similarity_search(query, k=3, filter=my_filter)
for d in found_docs:
  print(d.page_content)
  print(d.metadata["source"])
  print("---------------------------------------------------------------")

```