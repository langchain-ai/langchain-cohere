"""This module contains the tools that are used in the experiments."""
import os
import shutil
import tempfile
from typing import List, Optional, Union

import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool


def get_file_peek_tool(file_path: Union[str, List[str]]) -> Tool:
    file_name_path_map = (
        {os.path.split(file_path_ind)[1]: file_path_ind for file_path_ind in file_path}
        if isinstance(file_path, list)
        else {os.path.split(file_path)[1]: file_path}
    )

    def file_peek(filename: str, num_rows: int = 5) -> Union[pd.DataFrame, str]:
        """A function that will return a preview of the dataframe based on the table path.

        Args:
            table_path: the table path
            num_rows: the number of rows of the table to preview.
        """  # noqa E501
        file_path = file_name_path_map[filename]
        if ".csv" in filename:
            return pd.read_csv(file_path).head(num_rows).to_markdown()
        else:
            return "the table_path was not recognised"

    class file_peek_inputs(BaseModel):
        file_path: str = Field(description="the file path fo the csv to read!")

    file_peek_tool = Tool(
        name="file_peek",
        description="The name of the attached file to show a peek preview.",  # noqa E501
        func=file_peek,
        args_schema=file_peek_inputs,
    )

    return file_peek_tool


def get_python_tool(
    file_path: Union[str, List[str]],
    delete: bool = True,
    temp_dir: Optional[str] = None,
    temp_path_prefix: Optional[str] = "langchain",
    temp_path_suffix: Optional[str] = "csv_agent",
) -> Tool:
    if not temp_dir:
        temp_dir = tempfile.TemporaryDirectory(
            prefix=temp_path_prefix, suffix=temp_path_suffix
        ).name
    if isinstance(file_path, str):
        file_path = [file_path]
    for single_file_path in file_path:
        _, file_name = os.path.split(single_file_path)
        temp_path = os.path.join(temp_dir, file_name)
        shutil.copyfile(single_file_path, temp_path)

    def python_interpreter(code: str) -> str:
        """A function that will return the output of the python code.

        Args:
            code: the python code to run.
        """
        code = f"import os\nos.chdir('{temp_dir}')\n\n" + code
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
