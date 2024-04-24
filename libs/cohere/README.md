# Langchain-Cohere

This package contains the LangChain integrations for [Cohere](https://cohere.com/).

[Cohere](https://cohere.com/) empowers every developer and enterprise to build amazing products and capture true business value with language AI.

## Installation
- Install the `langchain-cohere` package:
```bash
pip install langchain-cohere
```

- Get a [Cohere API key](https://cohere.com/) and set it as an environment variable (`COHERE_API_KEY`)

## Migration from langchain-community

Cohere's integrations used to be part of the `langchain-community` package, but since version 0.0.30 the integration in `langchain-community` has been deprecated in favour `langchain-cohere`.

The two steps to migrate are:

1) Import from langchain_cohere instead of langchain_community, for example:
   * `from langchain_community.chat_models import ChatCohere` -> `from langchain_cohere import ChatCohere`
   * `from langchain_community.retrievers import CohereRagRetriever` -> `from langchain_cohere import CohereRagRetriever`
   * `from langchain.embeddings import CohereEmbeddings` -> `from langchain_cohere import CohereEmbeddings`
   * `from langchain.retrievers.document_compressors import CohereRerank` -> `from langchain_cohere import CohereRerank`

2) The Cohere Python SDK version is now managed by this package and only v5+ is supported.
   * There's no longer a need to specify cohere as a dependency in requirements.txt/pyproject.toml (etc.) 

## Supported LangChain Integrations

| API              | description                                         | Endpoint docs                                             | Import                                                                         | Example usage                                                                                                               |
|------------------|-----------------------------------------------------|-----------------------------------------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Chat             | Build chat bots                                     | [chat](https://docs.cohere.com/reference/chat)            | `from langchain_cohere import ChatCohere`                                      | [notebook](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/chat/cohere.ipynb)                  |
| RAG Retriever    | Connect to external data sources                    | [chat + rag](https://docs.cohere.com/reference/chat)      | `from langchain_cohere import CohereRagRetriever`                              | [notebook](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/cohere.ipynb)            |
| Text Embedding   | Embed strings to vectors                            | [embed](https://docs.cohere.com/reference/embed)          | `from langchain_cohere import CohereEmbeddings`                                | [notebook](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/text_embedding/cohere.ipynb)        |
| Rerank Retriever | Rank strings based on relevance                     | [rerank](https://docs.cohere.com/reference/rerank)        | `from langchain_cohere import CohereRerank`                                    | [notebook](https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/retrievers/cohere-reranker.ipynb)   |
| ReAct Agent      | Let the model choose a sequence of actions to take  | [chat + rag](https://docs.cohere.com/reference/chat)      | `from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent` | [notebook](https://github.com/cohere-ai/notebooks/blob/main/notebooks/Vanilla_Multi_Step_Tool_Use.ipynb)                    |


## Usage Examples

### Chat

```python
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage

llm = ChatCohere()

messages = [HumanMessage(content="Hello, can you introduce yourself?")]
print(llm.invoke(messages))
```

### ReAct Agent

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_cohere import ChatCohere, create_cohere_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor

llm = ChatCohere()

internet_search = TavilySearchResults(max_results=4)
internet_search.name = "internet_search"
internet_search.description = "Route a user query to the internet"

prompt = ChatPromptTemplate.from_template("{input}")

agent = create_cohere_react_agent(
    llm,
    [internet_search],
    prompt
)

agent_executor = AgentExecutor(agent=agent, tools=[internet_search], verbose=True)

agent_executor.invoke({
    "input": "In what year was the company that was founded as Sound of Music added to the S&P 500?",
})
```

### RAG Retriever

```python
from langchain_cohere import ChatCohere, CohereRagRetriever

rag = CohereRagRetriever(llm=ChatCohere())
print(rag.get_relevant_documents("Who are Cohere?"))
```

### Text Embedding

```python
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
print(embeddings.embed_documents(["This is a test document."]))
```

## Contributing

Contributions to this project are welcomed and appreciated.
The [LangChain contribution guide](https://python.langchain.com/docs/contributing/code/) has instructions on how to setup a local environment and contribute pull requests.
