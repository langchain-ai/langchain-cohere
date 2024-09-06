"""This module contains the tools that are used in the experiments."""

import pandas as pd
from langchain_core.tools import Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from pydantic import BaseModel, Field


def get_file_peek_tool() -> Tool:
    def file_peek(filename: str, num_rows: int = 5) -> str:
        """Returns the first textual contents of an uploaded file

        Args:
            table_path: the table path
            num_rows: the number of rows of the table to preview.
        """  # noqa E501
        if ".csv" in filename:
            return pd.read_csv(filename).head(num_rows).to_markdown()
        else:
            return "the table_path was not recognised"

    class file_peek_inputs(BaseModel):
        filename: str = Field(
            description="The name of the attached file to show a peek preview."
        )

    file_peek_tool = Tool(
        name="file_peek",
        description="The name of the attached file to show a peek preview.",  # noqa E501
        func=file_peek,
        args_schema=file_peek_inputs,
    )

    return file_peek_tool


def get_file_read_tool() -> Tool:
    def file_read(filename: str) -> str:
        """Returns the textual contents of an uploaded file, broken up in text chunks

        Args:
            filename (str): The name of the attached file to read.
        """  # noqa E501
        if ".csv" in filename:
            return pd.read_csv(filename).to_markdown()
        else:
            return "the table_path was not recognised"

    class file_read_inputs(BaseModel):
        filename: str = Field(description="The name of the attached file to read.")

    file_read_tool = Tool(
        name="file_read",
        description="Returns the textual contents of an uploaded file, broken up in text chunks",  # noqa E501
        func=file_read,
        args_schema=file_read_inputs,
    )

    return file_read_tool


def get_python_tool() -> Tool:
    """Returns a tool that will execute python code and return the output."""

    def python_interpreter(code: str) -> str:
        """A function that will return the output of the python code.

        Args:
            code: the python code to run.
        """
        return python_repl.run(code)

    python_repl = PythonAstREPLTool()
    python_tool = Tool(
        name="python_interpreter",
        description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",  # noqa E501
        func=python_interpreter,
    )

    class PythonToolInput(BaseModel):
        code: str = Field(description="Python code to execute.")

    python_tool.args_schema = PythonToolInput
    return python_tool
