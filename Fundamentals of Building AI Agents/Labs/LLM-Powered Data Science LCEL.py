#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# ## **DataWizard: AI-Powered Data Analysis**
# 

# Estimated time needed: **45** minutes
# 

# 
# 
# ## The challenge
# 
# In today's data-driven world, valuable insights are locked away in spreadsheets and datasets that most professionals don't have the technical skills to analyze. Business owners, managers, and domain experts know their data is valuable but lack the programming knowledge to extract meaningful insights.
# 
# ## About this lab
# 
# In this lab, you will create an AI-powered agent that can help non-technical users perform data science tasks through natural language. You will:
# 
# 1. Build a collection of LangChain tools that perform basic data science tasks:
#    - Listing available datasets
#    - Loading and analyzing CSV files
#    - Generating dataset summaries and statistics
#    - Training and evaluating machine learning models
# 
# 2. Document each tool's purpose and output format clearly to ensure the AI agent can use them effectively
# 
# 3. Demonstrate why regular conversational agents have limitations when working with structured data
# 
# 4. Implement an executor agent that can manage a multi-step workflow for data analysis
# 
# By the end of this lab, you'll understand how to combine language models with specialized tools to create practical applications that make data science accessible to everyone.
# 

# ## __Table of Contents__
# 
# <ol>
#     <li><a href="#Objectives">Objectives</a></li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
#             <li><a href="#Importing-required-libraries">Importing required libraries</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Langchain-tools">Langchain tools</a>
#         <ol>
#             <li><a href="#Dataset-caching-tool">Dataset caching tool</a></li>
#             <li><a href="#Summarization-tool">Summarization tool</a></li>
#             <li><a href="#DataFrame-method-execution-tool">DataFrame method execution tool</a></li>
#             <li><a href="#Model-evaluation-tools">Model evaluation tools</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Agents">Agents</a>
#         <ol>
#             <li><a href="#Agent-creation-and-limitations">Agent creation and limitations</a></li>
#             <li><a href="#Agent-executor-ReAct">Agent executor ReAct</a></li>
#         </ol>
#     </li>
# </ol>
# 
# <li><a href="#Authors">Authors</a></li>
# <li><a href="#Contributors">Contributors</a></li>
# 

# ## Objectives
# 
# By the end of this project, you will be able to:
# 1. **Create specialized LangChain tools**: Develop custom tools for performing key data science tasks like dataset discovery, analysis, and modeling.
# 2. **Implement efficient caching mechanisms**: Build a system that stores and manages datasets in memory to optimize performance across multiple queries.
# 3. **Design a natural language interface**: Connect a language model to your tools, enabling conversational access to data science capabilities.
# 4. **Develop context-aware agents**: Create an executor agent that can maintain state and execute multi-step data analysis workflows.
# 5. **Handle errors gracefully**: Implement robust error handling for file operations and data processing tasks.
# 6. **Test your assistant with real queries**: Evaluate the system with practical business questions that demonstrate its ability to make data science accessible.
# 
# This project equips you with the skills to bridge the gap between non-technical users and data insights, democratizing access to advanced analytics through conversation.
# 

# ----
# 

# ## Setup
# 

# For this lab, you will be using the following libraries:
# 
# * [`langchain`](https://python.langchain.com/docs/get_started/introduction) for building modular AI applications with tools and agents.
# * [`langchain-openai`](https://python.langchain.com/docs/integrations/llms/openai) for connecting LangChain with OpenAI's language models.
# * [`openai`](https://github.com/openai/openai-python) for accessing AI models that power your conversational interface.
# * [`pandas`](https://pandas.pydata.org/) for data manipulation and analysis of CSV datasets.
# * [`numpy`](https://numpy.org/) for numerical operations and array processing.
# * [`scikit-learn`](https://scikit-learn.org/stable/) for implementing machine learning models and evaluation metrics.
# * [`matplotlib`](https://matplotlib.org/) for creating data visualizations based on analysis results.
# * [`seaborn`](https://seaborn.pydata.org/) for enhanced statistical visualizations of dataset patterns.
# 

# ### Installing required libraries
# 
# The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You must run the following cell__ to install them. This step could take **several minutes**; please be patient.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/crvBKBOkg9aBzXZiwGEXbw/Restarting-the-Kernel.png" width="50%" alt="Restart kernel">
# 
# **NOTE**: If you encounter any issues, restart the kernel and run it again by clicking the **Restart the kernel** icon.
# 

# In[1]:


get_ipython().run_line_magic('pip', 'install langchain-openai==0.3.10 | tail -n 1')
get_ipython().run_line_magic('pip', 'install langchain==0.3.21 | tail -n 1')
get_ipython().run_line_magic('pip', 'install openai==1.68.2 | tail -n 1')
get_ipython().run_line_magic('pip', 'install pandas==2.2.3 | tail -n 1')
get_ipython().run_line_magic('pip', 'install numpy==2.2.4 | tail -n 1')
get_ipython().run_line_magic('pip', 'install matplotlib==3.10.1 | tail -n 1')
get_ipython().run_line_magic('pip', 'install seaborn==0.13.2 | tail -n 1')
get_ipython().run_line_magic('pip', 'install scikit-learn==1.6.1 | tail -n 1')


# Let's download the `.csv` files required in the project later.
# 

# In[1]:


get_ipython().system(' wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/N0CceRlquaf9q85PK759WQ/regression-dataset.csv')
get_ipython().system(' wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/7J73m6Nsz-vmojwab91gMA/classification-dataset.csv')


# ### Importing required libraries
# 
# Import all required libraries here:
# 

# In[2]:


import numpy as np
import pandas as pd
import matplotlib
import seaborn
import sklearn
import langchain
import openai
import langchain_openai

import glob 
import os
from typing import List, Optional


# # API Disclaimer
# This lab uses LLMs provided by OpenAI. This environment has been configured to allow LLM use without API keys so you can prompt them for **free (with limitations)**. With that in mind, if you wish to run this notebook **locally outside** of Skills Network's JupyterLab environment, you will have to configure your own API keys. Please note that using your own API keys means that you will incur personal charges.
# 
# ### Running Locally
# If you are running this lab locally, you will need to configure your own API key. This lab uses the `init_chat_model` function from `langchain`. To use the model you must set the environment variable `OPENAI_API_KEY` to your OpenAI API key. **DO NOT** run the cell below if you aren't running locally, it will causes errors.
# 

# In[4]:


# IGNORE IF YOU ARE NOT RUNNING LOCALLY
os.environ["OPENAI_API_KEY"] = "your OpenAI API key here"


# ## Langchain tools
# 
# Tools in LangChain are interfaces that allow an AI model (such as GPT-4) to interact with external systems, retrieve data, or perform actions beyond simple text generation. These tools act as APIs or function calls that the AI can invoke when needed.
# 
# First, you'll create a tool to identify available datasets in the local directory.  This tool helps your agent discover what CSV files are available for analysis without requiring the user to explicitly specify filenames. The assumption is that the CSV files have descriptive names that indicate their content. This is typically the first step in your data analysis workflow: discovering what data is available before loading or analyzing anything.
# 
# LangChain tools are simple Python functions wrapped with the `@tool` decorator (imported from `langchain_core.tools`). They allow Large Language Models (LLMs) to call specific functions, enabling structured workflows and external tool integrations.
# 
# ```python
# from langchain_core.tools import tool
# 
# @tool
# def my_tool(input: <type>) -> <output_type>:
#     """
#     Short description of what the tool does.
# 
#     Args:
#         input (<type>): Explanation of the input argument.
# 
#     Returns:
#         <output_type>: Explanation of the returned value.
#     """
#     # implement your logic here
#     return tool_output
# ```
# 
# The tool components:
# 
# - **`@tool` Decorator:** Marks the function as a tool, allowing LangChain to integrate it and expose it to the LLM.
# 
# - **Input Arguments:** The parameters your tool function accepts, along with type annotations for clarity.
# 
# - **Tool Description:** A clear, concise explanation used by LangChain and the LLM to understand when and how to call the tool.
# 
# - **Return Type:** Specifies the type of data your tool will return, improving clarity and reliability.
# 
# - **`.name`:** Automatically derived from your Python function name; used by LangChain to identify the tool.
# 
# - **`.description`:** Automatically extracted from your function's docstring; helps the LLM understand the tool’s purpose.
# 
# - **`.args`:** Represents input arguments with their associated types, allowing LangChain to validate and pass correct values to your tool function.
# 
# Let's create the first LangChain tool, which lists all CSV files in the current directory.
# 
# - `os.getcwd()` retrieves the current working directory.
# - `os.path.join(os.getcwd(), "*.csv")` constructs a path pattern to match all CSV files (`*` matches all filenames ending with `.csv`).
# - `glob.glob(pattern)` returns a list of files that match the given pattern.
# 
# 

# In[3]:


from langchain_core.tools import tool


@tool
def list_csv_files() -> Optional[List[str]]:
    """List all CSV file names in the local directory.

    Returns:
        A list containing CSV file names.
        If no CSV files are found, returns None.
    """
    csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
    if not csv_files:
        return None
    return [os.path.basename(file) for file in csv_files]


# You can print out the useful attributes of the tool, which helps during debugging and allows LangChain to identify the purpose and inputs of each function clearly:
# 

# In[4]:


print("Tool Name:", list_csv_files.name)
print("Tool Description:", list_csv_files.description)
print("Tool Arguments:", list_csv_files.args)


# ### Dataset caching tool
# 
#  As you build more complex tools, you need to efficiently manage datasets in memory. Since language models communicate via text, sending entire datasets in each response would waste tokens and context window space. To solve this, you'll create a global cache that stores DataFrames after they're first loaded. This approach has several benefits:
#    1. Reduces token usage by referencing datasets by name rather than content
#    2. Improves performance by loading data only once
#    3. Maintains dataset availability between different tool calls
# 
#  The following tool allows the agent to preload datasets into this cache system.
# 

# In[5]:


DATAFRAME_CACHE = {}

@tool
def preload_datasets(paths: List[str]) -> str:
    """
    Loads CSV files into a global cache if not already loaded.
    
    This function helps to efficiently manage datasets by loading them once
    and storing them in memory for future use. Without caching, you would
    waste tokens describing dataset contents repeatedly in agent responses.
    
    Args:
        paths: A list of file paths to CSV files.

    Returns:
        A message summarizing which datasets were loaded or already cached.
    """
    loaded = []
    cached = []
    for path in paths:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)
    
    return (
        f"Loaded datasets: {loaded}\n"
        f"Already cached: {cached}"
    )


# You may think that the `global` keyword would work effectively instead of `DATAFRAME CACHE`, but when using functions in Python with LangChain agents, simply writing the keyword `global` isn't enough to maintain data between different tool calls. This is because each time the agent runs a tool, it might do so in a separate execution environment, causing any `global` variables to reset. Instead, using a module-level dictionary (`DATAFRAME_CACHE`) that lives outside any function creates a persistent storage space that all tools can access without explicitly passing it around. This approach works reliably across multiple function calls, even when they happen in different contexts, and keeps the function interfaces clean by avoiding the need to pass the cache as an additional parameter to every tool.
#  
#  Well-structured docstrings are essential for your LLM-based tools because they serve as instructions for the AI agent. The agent relies on these descriptions to understand when and how to use each tool. Similarly, formatted outputs help the agent parse and interpret results properly. Always ensure your tool functions have clear docstrings and return well-structured outputs that are easy for both humans and AI to understand.
# 

# ### Summarization tool 
# Next, create a tool to provide dataset summaries with key statistical information. This tool gives the agent a quick overview of each dataset without transferring the entire content. It examines the structure of each CSV file and returns metadata that helps the agent understand what kinds of data it's working with.
# 
#  Type annotations are extremely important in the tools for several reasons:
#    1. They provide clear documentation of expected inputs and outputs
#    2. They enable static type checking to catch errors before runtime
#    3. Most critically, they help the AI agent understand exactly what data to provide and what format to expect in return
# 
# For this tool, nested type annotations (List[Dict[str, Any]]) precisely define the structured data that will be returned. This allows the agent to parse and utilize the results correctly in subsequent reasoning steps.
# 

# In[6]:


from typing import List, Optional,Dict,Any

@tool
def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze multiple CSV files and return metadata summaries for each.

    Args:
        dataset_paths (List[str]): 
            A list of file paths to CSV datasets.

    Returns:
        List[Dict[str, Any]]: 
            A list of summaries, one per dataset, each containing:
            - "file_name": The path of the dataset file.
            - "column_names": A list of column names in the dataset.
            - "data_types": A dictionary mapping column names to their data types (as strings).
    """
    summaries = []

    for path in dataset_paths:
        # Load and cache the dataset if not already cached
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
        
        df = DATAFRAME_CACHE[path]

        # Build summary
        summary = {
            "file_name": path,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        summaries.append(summary)

    return summaries


# ### DataFrame method execution tool
# 
# Now that you have a basic understanding of your datasets, you need a flexible way to explore and analyze them. Just like data scientists use various pandas DataFrame methods (`head()`, `describe()`, `info()`, etc.), your agent needs the ability to apply these methods to your cached datasets.
# 
# You'll leverage Python's `getattr()` function, which allows you to retrieve and call a method using its string name. This approach gives your agent the flexibility to select the most appropriate DataFrame method based on the current analysis needs.
# 
# By providing both the file name and the method name as parameters, the LLM can intelligently choose which analysis techniques to apply to different datasets. 
# The result of each method call is converted to a string representation, making it accessible to the LLM for further analysis and reasoning. 
# 

# In[7]:


@tool
def call_dataframe_method(file_name: str, method: str) -> str:
   """
   Execute a method on a DataFrame and return the result.
   This tool lets you run simple DataFrame methods like 'head', 'tail', or 'describe' 
   on a dataset that has already been loaded and cached using 'preload_datasets'.
   Args:
       file_name (str): The path or name of the dataset in the global cache.
       method (str): The name of the method to call on the DataFrame. Only no-argument 
                     methods are supported (e.g., 'head', 'describe', 'info').
   Returns:
       str: The output of the method as a formatted string, or an error message if 
            the dataset is not found or the method is invalid.
   Example:
       call_dataframe_method(file_name="data.csv", method="head")
   """
   # Try to get the DataFrame from cache, or load it if not already cached
   if file_name not in DATAFRAME_CACHE:
       try:
           DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
       except FileNotFoundError:
           return f"DataFrame '{file_name}' not found in cache or on disk."
       except Exception as e:
           return f"Error loading '{file_name}': {str(e)}"
   
   df = DATAFRAME_CACHE[file_name]
   func = getattr(df, method, None)
   if not callable(func):
       return f"'{method}' is not a valid method of DataFrame."
   try:
       result = func()
       return str(result)
   except Exception as e:
       return f"Error calling '{method}' on '{file_name}': {str(e)}"


# ### Model evaluation tools
# 
# These tools provide specialized functionality for building and evaluating machine learning models on datasets. The agent will first analyze the dataset structure using previous tools to determine if the prediction task is classification or regression. Then, based on its assessment, it will call either `evaluate_classification_dataset` or `evaluate_regression_dataset`, providing the appropriate dataset filename and target column. Both tools handle the technical aspects of machine learning (splitting the data, training the model, and calculating performance metrics) while abstracting away implementation details. For classification tasks, the agent will examine the target column's data type and unique values to determine if it's categorical, then call the classification evaluator, which returns accuracy metrics. For regression tasks involving continuous numerical predictions, the agent will select the regression evaluator that returns the R² score and mean squared error. This decision-making process showcases how a well-designed agent can choose the appropriate tool based on data characteristics, demonstrating intelligent workflow automation.
# 

# In[8]:


sklearn.model_selection import train_test_split
from sklearn.ensfromemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assumes this global cache is shared
DATAFRAME_CACHE = {}

@tool
def evaluate_classification_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    """
    Train and evaluate a classifier on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the classification target.
    Returns:
        Dict[str, float]: A dictionary with the model's accuracy score.
    """
    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
    
    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}

@tool
def evaluate_regression_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    """
    Train and evaluate a regression model on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the regression target.
    Returns:
        Dict[str, float]: A dictionary with R² score and Mean Squared Error.
    """
    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
    
    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
       "r2_score": r2,
         mse = mean_squared_error(y_test, y_pred)
    return {
    "mean_squared_error": mse
    }


# In[9]:


# ==============================
# Required Imports
# ==============================

import pandas as pd

from typing import Dict

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    r2_score,
    mean_squared_error
)

# Global dataset cache
DATAFRAME_CACHE = {}

# Dummy decorator if not defined
def tool(func):
    return func


# ======================================
# Classification Tool
# ======================================

@tool
def evaluate_classification_dataset(
    file_name: str,
    target_column: str
) -> Dict[str, float]:

    """
    Train classification model
    and return accuracy.
    """

    # Load dataset
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)

        except FileNotFoundError:
            return {
                "error": f"File '{file_name}' not found."
            }

        except Exception as e:
            return {
                "error": str(e)
            }

    df = DATAFRAME_CACHE[file_name]

    # Check column
    if target_column not in df.columns:
        return {
            "error": f"Column '{target_column}' not found."
        }

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        y_pred
    )

    return {
        "accuracy": float(acc)
    }


# ======================================
# Regression Tool
# ======================================

@tool
def evaluate_regression_dataset(
    file_name: str,
    target_column: str
) -> Dict[str, float]:

    """
    Train regression model
    and return R2 + MSE.
    """

    # Load dataset
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)

        except FileNotFoundError:
            return {
                "error": f"File '{file_name}' not found."
            }

        except Exception as e:
            return {
                "error": str(e)
            }

    df = DATAFRAME_CACHE[file_name]

    # Check column
    if target_column not in df.columns:
        return {
            "error": f"Column '{target_column}' not found."
        }

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Train regression model
    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    return {
        "r2_score": float(r2),
        "mean_squared_error": float(mse)
    }


# ## Agents
# 
# Agents in LangChain are advanced components that enable AI models to decide when and how to use tools dynamically. Instead of relying on predefined scripts, agents analyze user queries and choose the best tools to achieve a goal. The next step is defining your agent, which requires specifying how it should think and behave. You'll use `ChatPromptTemplate.from_messages()` to create a structured prompt with three essential components:
# 
# 1. **System message**: This establishes the agent's identity and primary objective. You define it as a data science assistant whose task is to analyze CSV files and determine whether each dataset is suitable for classification or regression based on its structure. This gives the agent a clear purpose and scope.
# 
# 2. **User input**: The `{input}` placeholder will be replaced with the user's actual query. This allows the agent to respond directly to what the user is asking about.
# 
# 3. **Agent scratchpad**: The `{agent_scratchpad}` placeholder is crucial for tool-calling agents as it provides space for the agent to show its reasoning process and track intermediate steps. This enables the agent to build a chain of thought, call tools sequentially, and use the results from one tool to inform decisions about subsequent tool calls.
# 
# ![agents copy.png](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/TYkDvBmpmmSXx6TftNpJgw/agents%20copy.png)
# 
# [Reference article for image](https://medium.com/@Shamimw/understanding-langchain-tools-and-agents-a-guide-to-building-smart-ai-applications-e81d200b3c12)
# 

# In[10]:


from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 🧠 Step 2: Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a data science assistant. Use the available tools to analyze CSV files. "
     "Your job is to determine whether each dataset is for classification or regression, based on its structure."),
    
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # Required for tool-calling agents
])


# Now, create a chatbot object
# 

# In[11]:


from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini", model_provider="openai", streaming=False )


# Create a list tool that contains all the tool objects
# 

# In[12]:


tools=[list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method, evaluate_classification_dataset, evaluate_regression_dataset]


# ### Agent creation and limitations
# 
# Here, you'll create your agent using `create_openai_tools_agent()`, which combines the language model, tools, and prompt template into a functional agent. However, this raw agent has significant limitations when used directly. It only performs a single step of reasoning and tool usage per invocation, then returns its intermediate thought process rather than a final answer. This behavior occurs because the agent doesn't automatically manage the full loop of thinking, acting, observing results, and continuing to reason until reaching a complete solution.
# 

# In[13]:


# Construct the tool calling agent
agent = create_openai_tools_agent(llm, tools, prompt)


# In[14]:


response = agent.invoke({
    "input": "Can you tell me about the dataset?",
    "intermediate_steps": []
})


# In[15]:


# Get the first ToolAgentAction from the list
action = response[0]

# Print the key details
print("🧠 Agent decided to call a tool:")
print("Tool Name:", action.tool)
print("Tool Input:", action.tool_input)
print("Log:\n", action.log.strip())


# When the agent was called with the input "Can you tell me about the dataset?", it responded with a tool action: it chose to invoke the list_csv_files tool without any arguments. It didn’t try to load or analyze the dataset 
# 
#  ReAct-style agents follow a step-by-step reasoning loop. ReAct stands for Reasoning and Acting: the agent thinks about what to do next, takes one action (like calling a tool), then waits for the result before continuing. This is why the agent's first instinct is to gather context—by listing the available CSV files—before attempting anything more complex. This isn’t a failure; it’s how the agent is designed to operate—reasoning one step at a time based on feedback. 
# 

# ### Agent executor ReAct
# 
# Managing this ReAct loop manually can be cumbersome, which is why you'll use the AgentExecutor. The AgentExecutor wraps the agent and the toolset, and handles the full tool-use loop behind the scenes. It automatically runs the agent, executes the selected tool, takes the result (observation), and feeds it back into the agent until a final answer is reached. Without the executor, you'd have to manually manage every step, including checking whether the agent returned a tool call or a final answer, running the tool, and tracking the intermediate steps—all of which the executor handles for you.
# 

# In[16]:


from langchain.agents import AgentExecutor


# #### Agent executor configuration
# 
# The `AgentExecutor` line creates a complete, autonomous agent system by wrapping your basic agent with additional functionality. This executor manages the full ReAct loop (reasoning and acting) that allows the agent to make multiple tool calls in sequence until reaching a final answer. Configure it with several important parameters:
# 
# 1. **agent**: The agent to run for creating a plan and determining actions to take at each step of the execution loop.
# 
# 2. **tools**: The valid tools the agent can call.
# 3. **verbose=True**: Enables detailed logging of each step in the agent's thinking and tool-calling process, which is invaluable for debugging and understanding how the agent arrives at its conclusions.
# 
# 4. **handle_parsing_errors=True**: Rather than crashing, the executor will attempt to recover and continue the conversation.
# 
# The second line, `agent_executor.agent.stream_runnable = False`, disables streaming mode for the agent. 
# 

# In[17]:


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True)
agent_executor.agent.stream_runnable = False


# You can now build a bot DataWizard:
# 

# In[ ]:


print("📊 Ask questions about your dataset (type 'exit' to quit):")

while True:
    user_input=input(" You:")
    if user_input.strip().lower() in ['exit','quit']:
        print("see ya later")
        break
        
    result=agent_executor.invoke({"input":user_input})
    print(f"my Agent: {result['output']}")


# ## Authors
# 

# [Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)
# 

# [Karan Goswami](https://author.skills.network/instructors/karan_goswami)
# 

# [Kunal Makwana](https://author.skills.network/instructors/kunal_makwana)
# 

# ## Contributors
# 

# [Wilbur Elbouni](https://author.skills.network/instructors/wilbur_elbouni)
# 

# Copyright © IBM Corporation. All rights reserved.
# 
