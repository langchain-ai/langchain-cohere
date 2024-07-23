import csv
import io
import os
from io import IOBase
from typing import List, Optional, Union

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.chat import (
    BaseMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.tools import BaseTool

from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.csv_agent.prompts import (
    CSV_PREAMBLE,
)

# lets define a set of tools for the Agent
from langchain_cohere.csv_agent.tools import get_file_peek_tool, get_python_tool


def create_prompt(
    system_message: Optional[BaseMessage] = SystemMessage(
        content="You are a helpful AI assistant."
    ),
    extra_prompt_messages: Optional[List[BaseMessagePromptTemplate]] = None,
) -> ChatPromptTemplate:
    """Create prompt for this agent.

    Args:
        system_message: Message to use as the system message that will be the
            first in the prompt.
        extra_prompt_messages: Prompt messages that will be placed between the
            system message and the new human input.

    Returns:
        A prompt template to pass into this agent.
    """
    _prompts = extra_prompt_messages or []
    messages: List[Union[BaseMessagePromptTemplate, BaseMessage]]
    if system_message:
        messages = [system_message]
    else:
        messages = []

    messages.extend(
        [
            *_prompts,
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    return ChatPromptTemplate(messages=messages)


def _get_csv_head(path: str, number_of_head_rows: int) -> str:
    with open(path, "r") as file:
        lines = []
        for _ in range(number_of_head_rows):
            lines.append(file.readline().split(","))
        # validate that the head contents are well formatted csv
        text = io.StringIO()
        writer = csv.writer(text)
        writer.writerow(lines)
        return text.getvalue()


def _get_prompt(
    path: Union[str, List[str]], number_of_head_rows: int
) -> ChatPromptTemplate:
    if isinstance(path, str):
        lines = _get_csv_head(path, number_of_head_rows)
        _, file_name = os.path.split(path)
        prompt_message = f"The user uploaded the following attachments:\nFilename: {file_name}\nWord Count: {count_words_in_file(path)}\nPreview: {' '.join(lines[:number_of_head_rows])}"  # noqa: E501

    elif isinstance(path, list):
        prompt_messages = []
        for file_path in path:
            lines = _get_csv_head(file_path, number_of_head_rows)
            prompt_messages.append(
                f"The user uploaded the following attachments:\nFilename: {file_path}\nWord Count: {count_words_in_file(file_path)}\nPreview: {' '.join(lines[:number_of_head_rows])}"  # noqa: E501
            )
        prompt_message = " ".join(prompt_messages)

    prompt = create_prompt(system_message=HumanMessage(prompt_message))
    return prompt


def count_words_in_file(file_path: str) -> int:
    try:
        with open(file_path, "r") as file:
            content = file.readlines()
            words = [len(sentence.split()) for sentence in content]
            return sum(words)
    except FileNotFoundError:
        print("File not found.")
        return 0
    except Exception as e:
        print("An error occurred:", str(e))
        return 0


def create_csv_agent(
    llm: BaseLanguageModel,
    path: Union[str, List[str]],
    extra_tools: List[BaseTool] = [],
    pandas_kwargs: Optional[dict] = None,
    prompt: Optional[ChatPromptTemplate] = None,
    number_of_head_rows: int = 5,
    verbose: bool = True,
    return_intermediate_steps: bool = True,
    temp_path_dir: Optional[str] = None,
    temp_path_prefix: Optional[str] = "langchain",
    temp_path_suffix: Optional[str] = "csv_agent",
) -> AgentExecutor:
    """Create csv agent with the specified language model.

    Args:
        llm: Language model to use for the agent.
        path: A string path, or a list of string paths
            that can be read in as pandas DataFrames with pd.read_csv().
        number_of_head_rows: Number of rows to display in the prompt for sample data
        include_df_in_prompt: Display the DataFrame sample values in the prompt.
        pandas_kwargs: Named arguments to pass to pd.read_csv().
        prefix: Prompt prefix string.
        suffix: Prompt suffix string.
        prompt: Prompt to use for the agent. This takes precedence over the other prompt arguments, such as suffix and prefix.
        temp_path_dir: Temporary directory to store the csv files in for the python repl.
        delete_temp_path: Whether to delete the temporary directory after the agent is done. This only works if temp_path_dir is not provided.

    Returns:
        An AgentExecutor with the specified agent_type agent and access to
        a PythonREPL and any user-provided extra_tools.

    Example:
        .. code-block:: python

            from langchain_cohere import ChatCohere, create_csv_agent

            llm = ChatCohere(model="command-r-plus", temperature=0)
            agent_executor = create_csv_agent(
                llm,
                "titanic.csv"
            )
            resp = agent_executor.invoke({"input":"How many people were on the titanic?"})
            print(resp.get("output"))
    """  # noqa: E501
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas package not found, please install with `pip install pandas`."
        )

    _kwargs = pandas_kwargs or {}
    if isinstance(path, (str)):
        df = pd.read_csv(path, **_kwargs)

    elif isinstance(path, list):
        df = []
        for item in path:
            if not isinstance(item, (str, IOBase)):
                raise ValueError(f"Expected str or file-like object, got {type(path)}")
            df.append(pd.read_csv(item, **_kwargs))
    else:
        raise ValueError(f"Expected str, list, or file-like object, got {type(path)}")

    if not prompt:
        prompt = _get_prompt(path, number_of_head_rows)

    final_tools = [
        get_python_tool(
            file_path=path,
            temp_dir=temp_path_dir,
            temp_path_prefix=temp_path_prefix,
            temp_path_suffix=temp_path_suffix,
        ),
        get_file_peek_tool(path),
    ] + extra_tools
    if "preamble" in llm.__dict__ and not llm.__dict__.get("preamble"):
        llm = ChatCohere(**llm.__dict__)
        llm.preamble = CSV_PREAMBLE

    agent = create_tool_calling_agent(llm=llm, tools=final_tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=final_tools,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
    )
    return agent_executor
