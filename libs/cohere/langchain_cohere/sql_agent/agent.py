"""Cohere SQL agent."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Optional,
    Sequence,
    Union,
)

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
)
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessage,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.sql_agent.prompts import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREAMBLE,
    SQL_PREFIX,
)

if TYPE_CHECKING:
    from langchain_classic.agents.agent import AgentExecutor
    from langchain_community.utilities.sql_database import SQLDatabase
    from langchain_core.callbacks import BaseCallbackManager
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.tools import BaseTool

from datetime import datetime

from langchain_classic.agents import (
    create_tool_calling_agent,
)
from langchain_classic.agents.agent import (
    AgentExecutor,
    RunnableMultiActionAgent,
)


def create_sql_agent(
    llm: BaseLanguageModel,
    toolkit: Optional[SQLDatabaseToolkit] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    extra_tools: Sequence[BaseTool] = (),
    *,
    db: Optional[SQLDatabase] = None,
    prompt: Optional[BasePromptTemplate] = None,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a SQL agent from an LLM and toolkit or database.

    Args:
        llm: Language model to use for the agent. If agent_type is "tool-calling" then
            llm is expected to support tool calling.
        toolkit: SQLDatabaseToolkit for the agent to use. Must provide exactly one of
            'toolkit' or 'db'. Specify 'toolkit' if you want to use a different model
            for the agent and the toolkit.
        callback_manager: DEPRECATED. Pass "callbacks" key into 'agent_executor_kwargs'
            instead to pass constructor callbacks to AgentExecutor.
        prefix: Prompt prefix string. Must contain variables "top_k" and "dialect".
        suffix: Prompt suffix string. Default depends on agent type.
        input_variables: DEPRECATED.
        top_k: Number of rows to query for by default.
        max_iterations: Passed to AgentExecutor init.
        max_execution_time: Passed to AgentExecutor init.
        early_stopping_method: Passed to AgentExecutor init.
        verbose: AgentExecutor verbosity.
        agent_executor_kwargs: Arbitrary additional AgentExecutor args.
        extra_tools: Additional tools to give to agent on top of the ones that come with
            SQLDatabaseToolkit.
        db: SQLDatabase from which to create a SQLDatabaseToolkit. Toolkit is created
            using 'db' and 'llm'. Must provide exactly one of 'db' or 'toolkit'.
        prompt: Complete agent prompt. prompt and {prefix, suffix, format_instructions,
            input_variables} are mutually exclusive.  Must contain variables "top_k" and "dialect".
            Can contain variables "table_info" or "table_names" if the prompt requires them.
        **kwargs: Arbitrary additional Agent args.

    Returns:
        An AgentExecutor with the specified agent_type agent.

    Example:

        .. code-block:: python

        from langchain_cohere import ChatCohere, create_sql_agent
        from langchain_community.utilities import SQLDatabase

        db = SQLDatabase.from_uri("sqlite:///Chinook.db")
        llm = ChatCohere(model="command-r-plus", temperature=0)
        agent_executor = create_sql_agent(llm, db=db, verbose=True)
        resp = agent_executor.run("Show me the first 5 rows of the 'Album' table.")
        print(resp.get("output"))

    """  # noqa: E501

    if toolkit is None and db is None:
        raise ValueError(
            "Must provide exactly one of 'toolkit' or 'db'. Received neither."
        )
    if toolkit and db:
        raise ValueError(
            "Must provide exactly one of 'toolkit' or 'db'. Received both."
        )

    toolkit = toolkit or SQLDatabaseToolkit(llm=llm, db=db)  # type: ignore[arg-type]
    tools = toolkit.get_tools() + list(extra_tools)
    if prompt is None:
        prefix = prefix or SQL_PREFIX
        prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)
        suffix = suffix or SQL_FUNCTIONS_SUFFIX
        # .bind params get overwritten by .bind_tools params
        if "preamble" in llm.__dict__ and not llm.__dict__.get("preamble"):
            preamble = SQL_PREAMBLE.format(
                dialect=toolkit.dialect,
                top_k=top_k,
                current_date=datetime.now().strftime("%A, %B %d, %Y %H:%M:%S"),
            )
            chat_cohere_args = {k: v for k, v in llm.__dict__.items() if v}
            chat_cohere_args["preamble"] = preamble

            llm = ChatCohere(**chat_cohere_args)
            sys_prompt = suffix
        else:
            # If llm is passed after .bind/.bind_tools, then preamble cannot be passed
            sys_prompt = prefix + "\n\n" + suffix
        messages: Sequence[
            Union[BaseMessage, HumanMessagePromptTemplate, MessagesPlaceholder]
        ] = [
            HumanMessage(content=sys_prompt),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

    else:
        if "top_k" in prompt.input_variables:
            prompt = prompt.partial(top_k=str(top_k))
        if "dialect" in prompt.input_variables:
            prompt = prompt.partial(dialect=toolkit.dialect)
        if any(key in prompt.input_variables for key in ["table_info", "table_names"]):
            db_context = toolkit.get_context()
            if "table_info" in prompt.input_variables:
                prompt = prompt.partial(table_info=db_context["table_info"])
                tools = [
                    tool for tool in tools if not isinstance(tool, InfoSQLDatabaseTool)
                ]
            if "table_names" in prompt.input_variables:
                prompt = prompt.partial(table_names=db_context["table_names"])
                tools = [
                    tool for tool in tools if not isinstance(tool, ListSQLDatabaseTool)
                ]

    runnable = create_tool_calling_agent(llm, tools, prompt)  # type: ignore
    agent = RunnableMultiActionAgent(  # type: ignore[assignment]
        runnable=runnable,
        input_keys_arg=["input"],
        return_keys_arg=["output"],
        **kwargs,
    )

    return AgentExecutor(
        name="Cohere SQL Agent Executor",
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )
