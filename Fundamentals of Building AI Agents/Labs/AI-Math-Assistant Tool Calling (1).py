#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Build an AI Math Assistant with LangChain Tool Calling**
# 
# Estimated time needed: **45** minutes
# 
# In this lab, you will learn how to build a simple agent with LangChain, enabling AI agents to perform specific tasks. You'll create a mathematical toolkit that allows an AI agent to perform basic arithmetic operations through natural language interaction. 
# 
# Through this lab, you'll build an agent that can understand and solve mathematical queries like "add 25 and 15, then multiply by 2" by breaking down complex operations into simple steps.
# 
# 

# # Table of Contents
# 
# 1. [Objectives](#Objectives)
# 2. [Setup](#Setup)
# 3. [Installing required libraries](#Installing-required-libraries)
# 4. [Loading the LLM: Choosing the right language model](#Loading-the-LLM:-Choosing-the-right-language-model)
# 5. [Function](#Function)
#    - 5.1 [Tool](#Tool)
#    - 5.2 [initialize_agent](#initialize_agent)
# 6. [Relationship between agent and LLM](#Relationship-between-agent-and-LLM)
# 7. [Key parameters of initialize_agent](#Key-parameters-of-initialize_agent)
# 8. [Orchestrating multiple tools with an agent: Mathematical toolkit](#Orchestrating-multiple-tools-with-an-agent:-Mathematical-toolkit)
#    - 8.1 [Subtraction tool](#Subtraction-tool)
# 9. [Building the agent](#Building-the-agent)
#    - 9.1 [Exploring LangChain's built-in tools](#Exploring-LangChain's-built-in-tools)
#    - 9.2 [Popular built-in tools](#Popular-built-in-tools)
#    - 9.3 [Example: Using the Wikipedia tool](#Example:-Using-the-Wikipedia-tool)
# 10. [Exercise: Create a power tool to calculate exponents](#Exercise:-Create-a-power-tool-to-calculate-exponents)
#     - 10.1 [Objective](#Objective)
#     - 10.2 [Step 1: Create the power tool](#Step-1:-Create-the-power-tool)
#     - 10.3 [Step 2: Create an agent with the power tool](#Step-2:-Create-an-agent-with-the-power-tool)
#     - 10.4 [Step 3: Test the agent](#Step-3:-Test-the-agent)
# 11. [Authors](#authors)
# 

# ## Objectives
# 
# After completing this lab, you will be able to:
# 
# - Explain the concept of tools in LangChain
# - Create custom tools for specific tasks
# - Build an AI agent that can use multiple tools
# - Debug and improve tool functionality
# - Test tool implementations with various inputs
# 

# ----
# 

# ## Setup
# 

# 
# For this lab, you will use the following libraries:
# 
# - **`langchain`**: For creating AI agents and tools
# - **`langchain.chat_models`**: For accessing language models
# - **`langchain.agents`**: For creating and managing AI agents
# 
# ---
# 

# ## Installing required libraries
# 
# The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You will need to run the following cell__ to install them:
# 

# In[1]:


get_ipython().run_line_magic('pip', 'install langchain==0.3.23 | tail -n 1')
get_ipython().run_line_magic('pip', 'install langchain-ibm==0.3.10 | tail -n 1')
get_ipython().run_line_magic('pip', 'install langchain-community==0.3.16 | tail -n 1')
get_ipython().run_line_magic('pip', 'install wikipedia==1.4.0 | tail -n 1')
get_ipython().run_line_magic('pip', 'install openai==1.77.0 | tail -n 1')
get_ipython().run_line_magic('pip', 'install langchain-openai==0.3.16 | tail -n 1')


# ## Import the required libraries
# 

# In[2]:


from langchain_ibm import ChatWatsonx
from langchain.agents import AgentType
import re


# ## Loading the LLM: Choosing the right language model
# 
# In this example, IBM’s `ChatWatsonx`will be used to load a language model (LLM) for interacting with tools. IBM’s models, like Granite 3.2 and Granite 3.3, are highly versatile and excel at advanced reasoning tasks.
# 
# That said, other providers offer LLMs with different strengths:
# 
# - **OpenAI (GPT-4/GPT-3.5)**: Best for versatility and advanced reasoning.
# - **Facebook (Meta, LLaMA)**: Open-access, highly customizable for specialized use cases.
# - **IBM watsonx Granite**: Ideal for enterprise applications with seamless integration.
# - **Anthropic (Claude)**: Focused on safety, reliability, and ethical AI.
# - **Cohere**: Affordable and efficient for lightweight, task-specific models.
# 
# ---
# 
# For this project, you'll use `ChatWatsonx` because:
# - It offers a simple API for quick setup.
# - It supports advanced configurations like:
#   - **`temperature`**: Adjusting randomness of responses.
#   - **`max_tokens`**: Limiting the length of responses.
# - IBM’s models are widely regarded as state-of-the-art for general-purpose reasoning and conversation.
# 

# In[3]:


llm = ChatWatsonx(
    model_id="ibm/granite-4-h-small",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
)


# Let's generate a simple response:
# 

# In[4]:


response = llm.invoke("What is tool calling in langchain?")
print("\nResponse Content: ", response.content)


# ## API Disclaimer
# This lab uses LLMs provided by **IBM watsonx.ai** and **OpenAI**. This environment has been configured to allow LLM use without API keys so you can prompt them for **free (with limitations)**. With that in mind, if you wish to run this notebook **locally outside** of Skills Network's JupyterLab environment, you will have to **configure your own API keys**. Please note that using your own API keys means that you will incur personal charges. 
# 

# ### Running Locally
# 
# If you are running this lab locally, you will need to configure your own API keys. This lab uses `ChatOpenAI` and `ChatWatsonx` modules from `langchain`. Both configurations are shown below with instructions. **Replace all instances** of both modules with the completed modules below throughout the lab. **DO NOT** run the cell below if you aren't running locally, it will causes errors.
# 
# IGNORE IF YOU ARE NOT RUNNING LOCALLY

from langchain_openai import ChatOpenAI
from langchain_ibm import ChatWatsonx

openai_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    api_key = "your openai api key here",
)

watsonx_llm = ChatWatsonx(
    model_id="ibm/granite-3-2-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="your project id associated with the API key",
    api_key="your watsonx.ai api key here",
)
# 
# 

# ## Function
# 
# In AI, a **tool** will call a basic **function** or capability that can be called on to perform a specific task. Think of it like a single item in a toolbox: just like a hammer, screwdriver, or wrench, an AI toolbox is full of specific functions designed to solve problems or get things done.
# 
# When building tools for tool calling, there are a few key principles to keep in mind:
# 
# 1. **Clear purpose**: Make sure the tool has a well-defined job.
# 2. **Standardized input**: The tool should accept input in a predictable, structured format so it’s easy to use.
# 3. **Consistent output**: Always return results in a format that’s easy to process or integrate with other systems.
# 4. **Comprehensive documentation**: Your tool should include clear, simple documentation that explains what it does, how to use it, and any quirks or limitations.
# 
# Remember, documentation isn’t just for other developers—it’s also for the language model (LLM) to understand the tool’s purpose and how to use it effectively.
# 
# For this example, you’ll start with a simple tool to add numbers. It’ll check off most of the basic requirements, but one key limitation is that it doesn’t handle **basic error cases**, like ignoring non-numeric input. Improving error handling will make the tool much more robust and ready for real-world use.
# 

# In[5]:


def add_numbers(inputs:str) -> dict:
    """
    Adds a list of numbers provided in the input dictionary or extracts numbers from a string.

    Parameters:
    - inputs (str): 
    string, it should contain numbers that can be extracted and summed.

    Returns:
    - dict: A dictionary with a single key "result" containing the sum of the numbers.

    Example Input (Dictionary):
    {"numbers": [10, 20, 30]}

    Example Input (String):
    "Add the numbers 10, 20, and 30."

    Example Output:
    {"result": 60}
    """
    numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]

    
    result = sum(numbers)
    return {"result": result}


# Directly testing a tool allows you to pinpoint where the problem lies—whether it’s in the tool’s logic, input parsing, or output formatting.
# Here, you'll input the string `1 2` and get the sum.
# 

# In[6]:


add_numbers("1 2") 


# 
# ## Tool
# The `Tool` class in LangChain serves as a structured wrapper that converts regular Python functions into agent-compatible tools. Each tool needs three key components:
# 1. A name that identifies the tool
# 2. The function that performs the actual operation
# 3. A description that helps the agent understand when to use the tool
# 
# Testing section improvement:
# 
# 
# 

# In[7]:


from langchain.agents import Tool
add_tool=Tool(
        name="AddTool",
        func=add_numbers,
        description="Adds a list of numbers and returns the result.")

print("tool object",add_tool)


# Let's see the parameters of the object:
# 
# - **`name`** (*str*):
#   - A unique identifier for the tool.
# 
#   - **Example**: `"AddTool"`
# 
# - **`.invoke`** (*Callable*):
#   - The function that the tool wraps.
#   - **Example**: `add_numbers`
# 
# - **`description`** (*str*):
#   - A concise explanation of what the tool does.
#   - **Example**: `"Adds a list of numbers and returns the result."`
# 
# 

# 
# These attributes allow you to inspect the tool object
# 

# In[8]:


# Tool name
print("Tool Name:")
print(add_tool.name)

# Tool description
print("Tool Description:")
print(add_tool.description)

# Tool function
print("Tool Function:")
print(add_tool.invoke)


# You can call the tool's function via the ```add_tool``` object:
# 

# In[9]:


print("Calling Tool Function:")
test_input = "10 20 30 a b" 
print(add_tool.invoke(test_input))  # Example


# Testing the tool object ensures:
# 
# 1. **The tool is correctly set up**:
#    - Metadata (`name`, `description`, etc.) is properly defined and aligns with its purpose.
#    - The function and schema (if applicable) are correctly configured.
# 
# 2. **The wrapped function behaves as expected**:
#    - The function performs the intended task correctly.
#    - It handles edge cases and invalid inputs gracefully.
# 
# 3. **The tool integrates smoothly with agents**:
#    - The tool's output aligns with what the agent expects.
#    - There are no compatibility issues when the agent calls the tool.
# 

# ### `@tool` operator
# 
# Now you know how to create a tool with a `Tool` class (using Tool Interface), there's actually another way that you can create a tool using `@tool` decorator. The recommended way to create tools is using the `@tool` decorator. This decorator is designed to simplify the process of tool creation and should be used in most cases. After defining a function, you can decorate it with `@tool` to create a tool that implements the Tool Interface.
# 
# `@tool` opertor makes tools out of functions. See below:
# 

# In[10]:


from langchain_core.tools import tool
import re

@tool
def add_numbers(inputs:str) -> dict:
    """
    Adds a list of numbers provided in the input string.
    Parameters:
    - inputs (str): 
    string, it should contain numbers that can be extracted and summed.
    Returns:
    - dict: A dictionary with a single key "result" containing the sum of the numbers.
    Example Input:
    "Add the numbers 10, 20, and 30."
    Example Output:
    {"result": 60}
    """
    # Use regular expressions to extract all numbers from the input
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    # numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]
    
    result = sum(numbers)
    return {"result": result}


# The above function will now act as a tool. You can inspect the tool's schema and other properties using:
# 

# In[11]:


print("Name: \n", add_numbers.name)
print("Description: \n", add_numbers.description) 
print("Args: \n", add_numbers.args) 


# You can call the tool using the ```invoke``` method.
# 

# In[12]:


test_input = "what is the sum between 10, 20 and 30 " 
print(add_numbers.invoke(test_input))  # Example


# 
# ### Use @tool-StructuredTool
# 
# The @tool decorator creates a StructuredTool with schema information extracted from function signatures and docstrings as show here. This helps LLMs better understand what inputs the tool expects and how to use it properly. While both approaches work, @tool is generally preferred for modern LangChain applications, especially with LangGraph and function-calling models.
# 

# In[13]:


# Comparing the two approaches
print("Tool Constructor Approach:")

print(f"Has Schema: {hasattr(add_tool, 'args_schema')}")
print("\n")

print("@tool Decorator Approach:")


print(f"Has Schema: {hasattr(add_numbers, 'args_schema')}")
print(f"Args Schema Info: {add_numbers.args}")


# In this example, the tool has two inputs: a string containing the numbers to add, and a second boolean input that determines whether to sum the absolute values of those numbers.
# 

# In[14]:


from typing import List

@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    """
    Adds a list of numbers provided as input.

    Parameters:
    - numbers (List[float]): A list of numbers to be summed.
    - absolute (bool): If True, use the absolute values of the numbers before summing.

    Returns:
    - float: The total sum of the numbers.
    """
    if absolute:
        numbers = [abs(n) for n in numbers]
    return sum(numbers)


# Let's compare the arguments for add_numbers_with_options and add_numbers. Both are structured tools. They both include the inputs field, which is a string input. However, add_numbers_with_options has an additional key-value pair: absolute, a boolean field with a default value of False. This means add_numbers_with_options supports optional behavior—taking the absolute value of the numbers—while add_numbers only handles basic numeric extraction and summation
# 

# In[15]:


print(f"Args Schema Info: {add_numbers_with_options.args}")
print(f"Args Schema Info: {add_numbers.args}")


# You can call the tool using a dictionary as input, where each key corresponds to a parameter name and the value is the input for that parameter. For example, to control whether the numbers are summed normally or with absolute values, set the absolute flag to False or True: You get -6 and 6, respectively 
# 

# In[16]:


print(add_numbers_with_options.invoke({"numbers":[-1.1,-2.1,-3.0],"absolute":False}))
print(add_numbers_with_options.invoke({"numbers":[-1.1,-2.1,-3.0],"absolute":True}))


# ## Improved tool return types with Python typing
# 
# When creating tools, you must accurately specify their return values. This helps the agent understand and handle different possible outputs.
# 
# 

# The function `sum_numbers_with_complex_output` returns a more flexible output format. It returns a dictionary containing a float value when numbers are successfully summed, or a descriptive error message as a string if no numbers are found or an issue occurs during processing.
# 

# In[17]:


from typing import Dict, Union

@tool
def sum_numbers_with_complex_output(inputs: str) -> Dict[str, Union[float, str]]:
    """
    Extracts and sums all integers and decimal numbers from the input string.

    Parameters:
    - inputs (str): A string that may contain numeric values.

    Returns:
    - dict: A dictionary with the key "result". If numbers are found, the value is their sum (float). 
            If no numbers are found or an error occurs, the value is a corresponding message (str).

    Example Input:
    "Add 10, 20.5, and -3."

    Example Output:
    {"result": 27.5}
    """
    matches = re.findall(r'-?\d+(?:\.\d+)?', inputs)
    if not matches:
        return {"result": "No numbers found in input."}
    try:
        numbers = [float(num) for num in matches]
        total = sum(numbers)
        return {"result": total}
    except Exception as e:
        return {"result": f"Error during summation: {str(e)}"}


# The function `sum_numbers_from_text` returns a straightforward output format. It extracts all integer values from the input string, sums them, and returns the total as a float. This function assumes that at least one valid number is present in the input and does not handle cases where no numbers are found or where an error might occur.
# 

# In[18]:


@tool
def sum_numbers_from_text(inputs: str) -> float:
    """
    Adds a list of numbers provided in the input string.
    
    Args:
        text: A string containing numbers that should be extracted and summed.
        
    Returns:
        The sum of all numbers found in the input.
    """
    # Use regular expressions to extract all numbers from the input
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    result = sum(numbers)
    return result


# ### `initialize_agent`
# 
# When you set up an agent, you're connecting tools and an LLM to work together seamlessly. The agent uses the LLM to understand what needs to be done and decides which tool to use based on the task. Here's a simple overview of the key parts:
# 
# 
# #### **Relationship between agent and LLM**
# - The **agent** acts as the decision-maker, figuring out which tools to use and when.
# - The **LLM** is the reasoning engine. It:
#   - Interprets the user's input.
#   - Helps the agent make decisions.
#   - Generates a response based on the output of the tools.
# 
# Think of the agent as the manager assigning tasks and the LLM as the brain solving problems or delegating work.
# 
# ---
# 
# #### **Key parameters of `initialize_agent`**
# 
# 1. **`tools`**- see above 
# 
# 2.  **`llm`** see above 
# 
# 3. **`agent`**:
#    - Specifies the reasoning framework for the agent.
#    - `"zero-shot-react-description"` enables:
#      - **Zero-shot reasoning**: The agent can solve tasks it hasn't seen before by thinking through the problem step by step.
#      - **React framework**: A logical loop of:
#        - **Reason** → Think about the task.
#        - **Act** → Use a tool to perform an action.
#        - **Observe** → Check the tool's output.
#        - **Plan** → Decide what to do next.
# 
# 4. **`verbose`**:
#    - If `True`, it prints detailed logs of the agent’s thought process.
#    - Useful for debugging or understanding how the agent makes decisions.
# 
# 
# 

# You can now create an agent object using initialize_agent.
# 

# In[19]:


from langchain.agents import initialize_agent

agent = initialize_agent([add_tool], llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)


# Now, you can run the agent by asking a question.
# 

# > When running an agent using .run() or .invoke(), you may occasionally encounter a situation where the code keeps executing indefinitely, even if the LLM has already produced a valid answer. This typically happens when the system encounters an OUTPUT_PARSING_ERROR — often due to formatting issues in the LLM's response.
# >
# > In such cases, the agent can get stuck in a loop and won’t terminate on its own. If you see this happening, simply click the stop button (■) in the top toolbar to interrupt execution.
# 

# In[20]:


# Use the agent
response =agent.run("In 2023, the US GDP was approximately $27.72 trillion, while Canada's was around $2.14 trillion and Mexico's was about $1.79 trillion what is the total.")


# In[21]:


response


# In[22]:


agent.invoke({"input": "Add 10, 20, two and 30"})


# The agent was asked to add the numbers 10, 20, "two," and 30. The agent first noticed that one of the inputs was the word "two" instead of a number, so it converted "two" to its numeric form, which is 2. After preparing the list of numbers (10, 20, 2, and 30), the agent decided to use the `AddTool` to perform the addition. It passed the numbers to the tool, which calculated the sum and returned the result as 62. Finally, the agent provided the answer: **62**.
# 

# #### **Structured chat zero shot react-description**
# 
# When selecting an agent in LangChain, two factors matter: the agent type and the tool format, especially the tool’s return type. Agents like zero-shot-react-description expect tools to take and return plain strings, which works well with manually defined Tool(...) wrappers. 
# 
# In contrast, structured agents like `structured-chat-zero-shot-react-description` or `openai-functions` are built to handle structured inputs and outputs via the @tool decorator. If a tool returns a dictionary but the agent expects a string, it can cause key errors or parsing failures. 
# 

# In the agent example below, use `sum_numbers_from_text` as a tool and `structured-chat-zero-shot-react-description` as the agent type. For the LLM, you'll use `Granite`.
# 

# In[ ]:


agent_2 = initialize_agent([sum_numbers_from_text], llm, agent="structured-chat-zero-shot-react-description", verbose=True, handle_parsing_errors=True)
response = agent_2.invoke({"input": "Add 10, 20 and 30"})
print(response)


# Now, for the below agent, you will be using `sum_numbers_with_complex_output` as the tool. As for the LLM, you are going to use `gpt-4.1-nano` and the agent type `openai-functions`. 
# 
# One thing to note here is Some LLMs, like `Granite`, cannot unpack dictionary outputs because they lack native support for structured output parsing. As a result, when you use `sum_numbers_with_complex_output` with the `structured-chat-zero-shot-react-description` agent type, the agent fails to interpret the returned dictionary and throws an input validation or parsing error.
# 

# In[23]:


from langchain_openai import ChatOpenAI

llm_ai = ChatOpenAI(model="gpt-4.1-nano")


# In[24]:


agent_3 = initialize_agent([sum_numbers_with_complex_output], llm_ai, agent="openai-functions", verbose=True, handle_parsing_errors=True)
response = agent_3.invoke({"input": "Add 10, 20 and 30"})
print(response)


# Now, let's move on to tools with multiple inputs. The agent below uses `Granite` as the LLM and `add_numbers_with_options` as the tool, which accepts multiple input parameters. However, if the tool returns a complex output—such as a dictionary like in `sum_numbers_with_complex_output`—you’ll need to switch to a model like GPT and use an agent type that supports both multi-input tools and structured outputs. Granite and similar models may not handle complex output parsing reliably, especially when used with agents like `structured-chat-zero-shot-react-description`.
# 

# In[25]:


agent_2 = initialize_agent(
    [add_numbers_with_options],
    llm,
    agent="structured-chat-zero-shot-react-description",
    verbose=True
)


# In[26]:


response = agent_2.invoke({
    "input": "Add -10, -20, and -30 using absolute values."
})
print(response)


# Let's try with OpenAI to see if it runs with multiple inputs.
# 

# In[27]:


agent_openai = initialize_agent(
    [add_numbers_with_options],
    llm_ai,
    agent="openai-functions",
    verbose=True
)


# In[28]:


response = agent_openai.invoke({
    "input": "Add -10, -20, and -30 using absolute values."
})
print(response)


# ### **`create_react_agent`**
# 
# As LangChain's `AgentExecutor` is being deprecated, create_react_agent from LangGraph provides a more flexible and powerful alternative for building AI agents. This function creates a graph-based agent that works with chat models and supports tool-calling functionality.
# 
# ---
# 
# #### **Key parameters of `create_react_agent`**
# 
# 1. **`model`**
#     - The language model that powers the agent's reasoning.
#     - Must support tool calling for full functionality.
# 
# 2.  **`tools`**
#     - A list of tools the agent can use to perform actions.
#     - Can be LangChain tools, Python functions with @tool decorator, or a ToolNode instance
#     - Each tool should have a name, description, and implementation
# 
# 3. **`prompt (optional)`**:
#    - Customizes the instructions given to the LLM
#    - Can be:
#         - A string (converted to a SystemMessage)
#         - A SystemMessage object
#         - A function that transforms the state
#         - A Runnable that processes the state
# 
# and other parameters. To see more parameters, see [docs](https://langchain-ai.github.io/langgraph/reference/prebuilt/).
# 
# #### How it works
# 
# Unlike the legacy `AgentExecutor`, which used a fixed loop structure, `create_react_agent` creates a graph with these key nodes:
# 
# 1. **Agent Node:** Calls the LLM with the message history
# 2. **Tools Node:** Executes any tool calls from the LLM's response
# 3. **Continue/End Nodes:** Manage the workflow based on whether tool calls are present
# 
# The graph follows this process:
# 
# 1. User message enters the graph
# 2. LLM generates a response, potentially with tool calls
# 3. If tool calls exist, they're executed and their results are added to the message history
# 4. The updated messages are sent back to the LLM
# 5. This loop continues until the LLM responds without tool calls
# 6. The final state with all messages is returned
# 

# In[29]:


get_ipython().run_line_magic('pip', 'install langgraph==0.6.1 | tail -n 1')


# In[30]:


from langgraph.prebuilt import create_react_agent

agent_exec = create_react_agent(model=llm_ai, tools=[sum_numbers_from_text])
msgs = agent_exec.invoke({"messages": [("human", "Add the numbers -10, -20, -30")]})


# In[31]:


print(msgs["messages"][-1].content)


# ## Orchestrating multiple tools with an agent: Mathematical toolkit
# In real-world applications, a single tool is often insufficient to address the complexity and diversity of user requests. Tasks such as data analysis, performing calculations, or retrieving specific information require specialized capabilities that cannot be fulfilled by a single function. By equipping an agent with multiple tools, each tailored to a distinct purpose, you'll create a system that can dynamically select and utilize the appropriate tool based on the user’s query. This approach enhances the flexibility and scalability of the AI, allowing it to handle a broad spectrum of tasks with precision and efficiency. The orchestration of multiple tools ensures that the agent can seamlessly manage complex workflows, making it an essential framework for building robust and versatile AI systems.
# 
# To demonstrate this concept, let’s create additional tools, i.e, a mathematical toolkit. In addition to the addition tool, you will now create tools for subtraction, multiplication, and division. These tools will be integrated into an agent capable of handling various mathematical queries, showcasing how multiple tools can work together within a single AI system.
# 
# ### Subtraction tool
# The subtraction tool is designed to take a list of numbers and return the result of subtracting all subsequent numbers from the first number. This tool is particularly useful for handling queries involving differences, such as "What is 100 minus 20 and then minus 10?". 
# 

# In[32]:


@tool
def subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string, negates the first number, and successively subtracts 
    the remaining numbers in the list.

    This function is designed to handle input in string format, where numbers are separated 
    by spaces, commas, or other delimiters. It parses the string, extracts valid numeric values, 
    and performs a step-by-step subtraction operation starting with the first number negated.

    Parameters:
    - inputs (str): 
      A string containing numbers to subtract. The string may include spaces, commas, or 
      other delimiters between the numbers.

    Returns:
    - dict: 
      A dictionary containing the key "result" with the calculated difference as its value. 
      If no valid numbers are found in the input string, the result defaults to 0.

    Example Input:
    "100, 20, 10"

    Example Output:
    {"result": -130}

    Notes:
    - Non-numeric characters in the input are ignored.
    - If the input string contains only one valid number, the result will be that number negated.
    - Handles a variety of delimiters (e.g., spaces, commas) but does not validate input formats 
      beyond extracting numeric values.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]

    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Start with the first number negated
    result = -1 * numbers[0]

    # Subtract all subsequent numbers
    for num in numbers[1:]:
        result -= num

    return {"result": result}


# You can inspect the tool's schema and other properties using:
# 

# In[33]:


print("Name: \n", subtract_numbers.name)
print("Description: \n", subtract_numbers.description) 
print("Args: \n", subtract_numbers.args) 


# Let's use it directly by calling the function:
# 

# In[34]:


print("Calling Tool Function:")
test_input = "10 20 30 and four a b" 
print(subtract_numbers.invoke(test_input))  # Example


# Let's now build multiple tools, starting with `MultiplyTool` and `DivideTool`, by defining two functions: `multiply_numbers` and `divide_numbers`. These functions are simple - `multiply_numbers` takes a list of numbers in string format and returns their product, while `divide_numbers` takes the first number and divides it by each subsequent number in sequence. Instead of manually wrapping these functions in the Tool class, you'll use the `@tool` decorator to automatically convert them into LangChain tools, using their docstrings as descriptions. These decorated tools can then be added directly to the agent alongside other operations like addition or subtraction, allowing the agent to intelligently select the appropriate operation based on the user's query, making it versatile for handling various math problems.
# 

# In[35]:


# Multiplication Tool
@tool
def multiply_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates their product.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: A dictionary with the key "result" containing the product of the numbers.

    Example Input:
    "2, 3, 4"

    Example Output:
    {"result": 24}

    Notes:
    - If no numbers are found, the result defaults to 1 (neutral element for multiplication).
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    print(numbers)

    # If no numbers are found, return 1
    if not numbers:
        return {"result": 1}

    # Calculate the product of the numbers
    result = 1
    for num in numbers:
        result *= num
        print(num)

    return {"result": result}


# In[36]:


# Division Tool
@tool
def divide_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates the result of dividing the first number 
    by the subsequent numbers in sequence.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: A dictionary with the key "result" containing the quotient.

    Example Input:
    "100, 5, 2"

    Example Output:
    {"result": 10.0}

    Notes:
    - If no numbers are found, the result defaults to 0.
    - Division by zero will raise an error.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]


    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Calculate the result of dividing the first number by subsequent numbers
    result = numbers[0]
    for num in numbers[1:]:
        result /= num

    return {"result": result}


# When testing these mathematical tools directly, notice that using raw string inputs like "2, 3, and four" or "100, 5, two" will fail. The tools are designed to work with numeric inputs only - they don't have the natural language understanding that comes with the LLM agent layer. To test properly, you need to use a numeric value:
# 

# In[37]:


# Testing multiply_tool
multiply_test_input = "2, 3, and four "
multiply_result = multiply_numbers.invoke(multiply_test_input)
print("--- Testing MultiplyTool ---")
print(f"Input: {multiply_test_input}")
print(f"Output: {multiply_result}")


# In[38]:


# Testing divide_tool
divide_test_input = "100, 5, two"
divide_result = divide_numbers.invoke(divide_test_input)
print("--- Testing DivideTool ---")
print(f"Input: {divide_test_input}")
print(f"Output: {divide_result}")


# ## Building the agent
# 
# With the implementation of mathematical operators—addition, subtraction, multiplication, and division — you have established a simple yet functional mathematical toolkit. Unlike before, the agent must now not only select the appropriate tool and process the input but also determine the correct mathematical operation based on the user's query.
# 
# Let's create the agent object. first, combine all tools into a single list:
# 

# In[39]:


tools = [add_numbers,subtract_numbers, multiply_numbers, divide_numbers]
tools


# Like before, you will create the agent with the tools and the language model as input.
# 

# In[40]:


from langgraph.prebuilt import create_react_agent

# Create the agent with all tools
math_agent = create_react_agent(
    model=llm_ai,
    tools=tools,
    # Optional: Add a system message to guide the agent's behavior
    prompt="You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly."
)


# In[41]:


response = math_agent.invoke({
    "messages": [("human", "What is 25 divided by 4?")]
})

# Get the final answer
final_answer = response["messages"][-1].content
print(final_answer)


# In[42]:


response_2 = math_agent.invoke({
    "messages": [("human", "Subtract 100, 20, and 10.")]
})

# Get the final answer
final_answer_2 = response_2["messages"][-2].content
print(final_answer_2)


# When the agent tries to subtract 20 and 10 from 100, something unexpected happens. The tool called `SubtractTool` works differently than the agent expected. When you type in "100, 20, 10", instead of giving you 70 like you'd expect, it gives you -130. This happens because your special calculator first turns 100 into -100, then subtracts the other numbers.
# 
# ```The Confusion``` 
# 
# The agent expects the function to work like normal math 100 - 20 - 10 = 70). When the agent tries to fix this by breaking the problem into smaller steps, it still gets unexpected answers because the calculator keeps using its special rules.
# 
# 
# ```Getting Stuck```
# 
#  The agent keeps trying the same approach repeatedly, not realizing why it isn't working. Eventually, it runs out of time without solving the problem.
#  
#  Before you fix the problem, let's test the other tools.
# 

# In[43]:


print("\n--- Testing MultiplyTool ---")
response = math_agent.invoke({
    "messages": [("human", "Multiply 2, 3, and four.")]
})
print("Agent Response:", response["messages"][-1].content)

print("\n--- Testing DivideTool ---")
response = math_agent.invoke({
    "messages": [("human", "Divide 100 by 5 and then by 2.")]
})
print("Agent Response:", response["messages"][-1].content)


# Now lets change the `SubtractTool` so it subtracts the numbers directly (without negating the first number). This aligns the tool’s behavior with standard arithmetic and the agent’s expectations.
# 

# In[44]:


@tool
def new_subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and performs subtraction sequentially, starting with the first number.

    This function is designed to handle input in string format, where numbers may be separated by spaces, 
    commas, or other delimiters. It parses the input string, extracts numeric values, and calculates 
    the result by subtracting each subsequent number from the first. inputs[0]-inputs[1]-inputs[2]

    Parameters:
    - inputs (str): 
      A string containing numbers to subtract. The string can include spaces, commas, or other 
      delimiters between the numbers.

    Returns:
    - dict: 
      A dictionary containing the key "result" with the calculated difference as its value. 
      If no valid numbers are found in the input string, the result defaults to 0.

    Example Usage:
    - Input: "100, 20, 10"
    - Output: {"result": 70}

    Limitations:
    - The function does not handle cases where numbers are formatted with decimals or other non-integer representations.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]

    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Start with the first number
    result = numbers[0]

    # Subtract all subsequent numbers
    for num in numbers[1:]:
        result -= num

    return {"result": result}


# 
# ## Note: Tool naming when demonstrating different approaches
# 
# In this lab, two different ways to create the same mathematical tool (addition) are intentionally shown:
# 
# 1. Using the `Tool()` constructor approach (`add_tool`)
# 2. Using the `@tool` decorator approach (`add_numbers`)
# 
# This is to compare different LangChain tool creation methods for educational purposes. In a production application, you would typically choose one consistent approach rather than having duplicate tools for the same functionality.
# 
# When building real agents, duplicate tools with similar functions will confuse the LLM, as it won't know which one to choose. Always use unique tools with clearly differentiated purposes in production code.
# 

# Next, create a new agent, ensuring it incorporates the updated subtraction tool.
# 

# In[45]:


tools_updated = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers]
# Create the agent with all tools
math_agent_new = create_react_agent(
    model=llm,
    tools=tools_updated,
    # Optional: Add a system message to guide the agent's behavior
    prompt="You are a helpful mathematical assistant that can perform various operations. Use the tools precisely and explain your reasoning clearly."
)
print("agent",math_agent_new)


# Now, you are going to create a Python dictionary to test multiple use cases for your agent. Testing your agent is important on its own because it helps ensure that it works correctly in different situations. Automating test cases makes this process easier and helps catch errors before they become a problem. A good test suite checks how the agent handles different inputs, including tricky cases like dividing by zero, working with large numbers, and handling decimals. You can also test how the agent deals with mixed operations, like combining addition and multiplication.
# 

# In[46]:


# Test Cases
test_cases = [
    {
        "query": "Subtract 100, 20, and 10.",
        "expected": {"result": 70},
        "description": "Testing subtraction tool with sequential subtraction."
    },
    {
        "query": "Multiply 2, 3, and 4.",
        "expected": {"result": 24},
        "description": "Testing multiplication tool for a list of numbers."
    },
    {
        "query": "Divide 100 by 5 and then by 2.",
        "expected": {"result": 10.0},
        "description": "Testing division tool with sequential division."
    },
    {
        "query": "Subtract 50 from 20.",
        "expected": {"result": -30},
        "description": "Testing subtraction tool with negative results."
    }

]


# This code extracts the actual computation result from the agent's response structure. Unlike a direct tool invocation that returns a simple dictionary, LangGraph agents return a complex structure containing the entire conversation history as a list of messages. To find the computation result, you must locate the specific ToolMessage in this list (identified by its name matching one of the math tools), then parse its content, which contains the actual result as a JSON string. This approach is necessary because the result isn't accessible directly from the response object but is instead nested within the message history, requiring you to navigate through the messages to find and extract the relevant data for comparison with your expected values.
# 

# In[47]:


correct_tasks = []
# Corrected test execution
for index, test in enumerate(test_cases, start=1):
    query = test["query"]
    expected_result = test["expected"]["result"]  # Extract just the value
    
    print(f"\n--- Test Case {index}: {test['description']} ---")
    print(f"Query: {query}")
    
    # Properly format the input
    response = math_agent_new.invoke({"messages": [("human", query)]})
    
    # Find the tool message in the response
    tool_message = None
    for msg in response["messages"]:
        if hasattr(msg, 'name') and msg.name in ['add_numbers', 'new_subtract_numbers', 'multiply_numbers', 'divide_numbers']:
            tool_message = msg
            break
    
    if tool_message:
        # Parse the tool result from its content
        import json
        tool_result = json.loads(tool_message.content)["result"]
        print(f"Tool Result: {tool_result}")
        print(f"Expected Result: {expected_result}")
        
        if tool_result == expected_result:
            print(f"✅ Test Passed: {test['description']}")
            correct_tasks.append(test["description"])
        else:
            print(f"❌ Test Failed: {test['description']}")
    else:
        print("❌ No tool was called by the agent")

print("\nCorrectly passed tests:", correct_tasks)


# The current functions would benefit from enhanced error handling and input validation. The add_numbers, subtract_numbers, multiply_numbers, and divide_numbers functions should be updated to handle decimal numbers using float conversion, validate inputs more strictly, and provide clear error messages for edge cases. For example, divide_numbers should explicitly check for division by zero, and all functions should gracefully handle non-numeric inputs like "two" or "hundred". The test cases should be expanded beyond basic operations to include edge cases like divided by zero, empty inputs, and mixed numeric/text inputs (e.g., "divide one hundred by 5"). Also consider adding tests for decimal numbers (e.g., "multiply 3.5 by 2") and sequential operations (e.g., "multiply 10 by 2 then add 5"). This comprehensive testing approach ensures the agent can handle a wide range of real-world mathematical queries.
# 

# ## **Exploring LangChain's built-in tools**
# 
# While creating custom tools is powerful, LangChain provides a rich ecosystem of **pre-built tools** that solve common tasks out of the box. These tools abstract away complex implementation details (API calls, input parsing, error handling) and let you focus on building robust agents quickly.
# 

# ---
# 
# #### **Why use built-in tools?**
# - **Reliability**: Tested and maintained by the LangChain community.
# - **Time-saving**: No need to reinvent the wheel for common tasks.
# - **Integration**: Designed to work seamlessly with LangChain agents.
# 
# ---
# 

# #### **Popular built-in tools**
# Here are some widely used tools from `langchain_community.tools`:
# 
# | Tool Name               | Description                                                                 |
# |-------------------------|-----------------------------------------------------------------------------|
# | `WikipediaQueryRun`     | Search Wikipedia for factual information.                                   |
# | `GoogleSearchRun`       | Perform web searches using Google’s API.                                    |
# | `PythonREPLTool`        | Execute Python code in a safe environment.                                  |
# | `OpenWeatherMapQueryRun`| Fetch real-time weather data.                                               |
# | `YouTubeSearchTool`     | Search for YouTube videos.                                                  |
# 
# ---
# 

# 
# #### **Example: Using the Wikipedia tool**
# Let’s enhance the math agent with Wikipedia access to answer questions requiring factual context.
# 

# Now, you'll start by creating a Wikipedia search tool using `@tool` operator. This tool will allow the agent to fetch factual information from Wikipedia when needed.
# 

# In[48]:


from langchain_community.utilities import WikipediaAPIWrapper

# Create a Wikipedia tool using the @tool decorator
@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information about a topic.
    
    Parameters:
    - query (str): The topic or question to search for on Wikipedia
    
    Returns:
    - str: A summary of relevant information from Wikipedia
    """
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)


# In[49]:


search_wikipedia.invoke("What is tool calling?")


# Now, you will **create a list of available tools** (both custom math tools and a using wikipedia tool `search_wikipedia`) and then **initialize an agent** that can use these tools to solve problems. This agent will combine:  
# - Custom math tools (`add_numbers`, `new_subtract_numbers`, etc.) for arithmetic operations.  
# - A built-in tool (`wiki_tool`, e.g., for Wikipedia searches) for additional functionality.  
# 
# By combining these tools, the agent can handle **both mathematical calculations** (e.g., addition, subtraction) and **informational queries** (e.g., fetching facts from Wikipedia), depending on the user’s request.
# 

# In[50]:


# Update your tools list to include the Wikipedia tool
tools_updated = [add_numbers, new_subtract_numbers, multiply_numbers, divide_numbers, search_wikipedia]

# Create the agent with all tools including Wikipedia
math_agent_updated = create_react_agent(
    model=llm_ai,
    tools=tools_updated,
    prompt="You are a helpful assistant that can perform various mathematical operations and look up information. Use the tools precisely and explain your reasoning clearly."
)


# Now, you will **ask the agent a multi-step question** that requires:  
# 1. **Online searching** (using `search_wikipedia` or another built-in tool) to fetch real-world data.  
# 2. **Mathematical computation** (using `multiply_numbers`) to process the retrieved data.  
# 
# 
# 

# In[51]:


query = "What is the population of Canada? Multiply it by 0.75"

response = math_agent_updated.invoke({"messages": [("human", query)]})

print("\nMessage sequence:")
for i, msg in enumerate(response["messages"]):
    print(f"\n--- Message {i+1} ---")
    print(f"Type: {type(msg).__name__}")
    if hasattr(msg, 'content'):
        print(f"Content: {msg.content}")
    if hasattr(msg, 'name'):
        print(f"Name: {msg.name}")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        print(f"Tool calls: {msg.tool_calls}")


# **How it works**:
# 1. The agent first uses `search_wikipedia` to find Canada's population.
# 2. Extracts the numeric value from Wikipedia’s response.
# 3. Uses `multiply_numbers` to calculate 75% of the population.
# 4. Returns the final result with context.
# 

# For a full list of available tools, see the [LangChain Tools Documentation](https://python.langchain.com/docs/integrations/tools/).
# 

# ## **Exercise: Create a power tool to calculate exponents**
# 
# #### **Objective**
# In this exercise, you will create a custom tool that calculates the power of a number (e.g., \( x^y \)). You will then integrate this tool into an agent and test its functionality.
# 
# ---
# 
# #### **Step 1: Create the power tool**
# 
# 1. **Define the Tool Function**:
#    - Create a Python function named `calculate_power` that takes a string as input. The string will contain two numbers: the base (\( x \)) and the exponent (\( y \)).
#    - The function should extract the numbers, calculate \( x^y \), and return the result as a dictionary with the key `"result"`.
# 

# In[53]:


#TODO
def calculate_power(input_text: str) -> dict:
    """
    Calculates the power of a number (x^y).

    Parameters:
    - input_text (str): A string like "2, 3", "2 3", "5^2", or "2 to the power of 3".

    Returns:
    - dict: {"result": <calculated value>} or an error message.
    """
    # Try to extract expressions like "5^2"
    match = re.search(r"(\d+(?:\.\d+)?)\s*\^+\s*(\d+(?:\.\d+)?)", input_text)
    if match:
        base = float(match.group(1))
        exponent = float(match.group(2))
        return {"result": base ** exponent}

    # Try to extract expressions like "2 to the power of 3"
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to\s+the\s+power\s+of)\s*(\d+(?:\.\d+)?)", input_text, re.IGNORECASE)
    if match:
        base = float(match.group(1))
        exponent = float(match.group(2))
        return {"result": base ** exponent}

    # Fallback: assume two numbers separated by space or comma
    try:
        numbers = [float(num) for num in input_text.replace(",", " ").split()]
        if len(numbers) != 2:
            return {"result": "Invalid input. Please provide exactly two numbers."}
        base, exponent = numbers
        return {"result": base ** exponent}
    except ValueError:
        return {"result": "Invalid input format. Provide input like '2 3', '2^3', or '2 to the power of 3'."}


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# def calculate_power(input_text: str) -> dict:
#     """
#     Calculates the power of a number (x^y).
# 
#     Parameters:
#     - input_text (str): A string like "2, 3", "2 3", "5^2", or "2 to the power of 3".
# 
#     Returns:
#     - dict: {"result": <calculated value>} or an error message.
#     """
#     # Try to extract expressions like "5^2"
#     match = re.search(r"(\d+(?:\.\d+)?)\s*\^+\s*(\d+(?:\.\d+)?)", input_text)
#     if match:
#         base = float(match.group(1))
#         exponent = float(match.group(2))
#         return {"result": base ** exponent}
# 
#     # Try to extract expressions like "2 to the power of 3"
#     match = re.search(r"(\d+(?:\.\d+)?)\s*(?:to\s+the\s+power\s+of)\s*(\d+(?:\.\d+)?)", input_text, re.IGNORECASE)
#     if match:
#         base = float(match.group(1))
#         exponent = float(match.group(2))
#         return {"result": base ** exponent}
# 
#     # Fallback: assume two numbers separated by space or comma
#     try:
#         numbers = [float(num) for num in input_text.replace(",", " ").split()]
#         if len(numbers) != 2:
#             return {"result": "Invalid input. Please provide exactly two numbers."}
#         base, exponent = numbers
#         return {"result": base ** exponent}
#     except ValueError:
#         return {"result": "Invalid input format. Provide input like '2 3', '2^3', or '2 to the power of 3'."}
# 
# 
# ```
# 
# </details>
# 

# 2. **Create the tool object**:
#    - Use the `Tool` class from LangChain to create a tool object for the `calculate_power` function.
#    - Provide a name, description, and the function to the tool.
# 

# In[ ]:


#TODO
power_tool = Tool(
   name="PowerTool",
   func=calculate_power,
   description="Calculates the power of a number (x^y). Input should be two numbers: base and exponent."
)


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# power_tool = Tool(
#    name="PowerTool",
#    func=calculate_power,
#    description="Calculates the power of a number (x^y). Input should be two numbers: base and exponent."
# )
# ```
# 
# </details>
# 

# #### **Step 2: Create an agent with the power tool**
# 
# 1. **Set up the agent**:
#    - Use the `initialize_agent` function from LangChain to create an agent.
#    - Include the `power_tool` in the list of tools provided to the agent.
#    - Specify the agent type (e.g., `zero-shot-react-description`).
# 

# In[ ]:


#TODO
# List of tools for the agent
tools = [power_tool]

# Create the agent
agent = initialize_agent(
   tools,
   llm,
   agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
   verbose=True,
    handle_parsing_errors=True
)


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# # List of tools for the agent
# tools = [power_tool]
# 
# # Create the agent
# agent = initialize_agent(
#    tools,
#    llm,
#    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#    verbose=True,
#     handle_parsing_errors=True
# )
# ```
# 
# </details>
# 

# #### **Step 3: Test the agent**
# 
# 1. **Test the Agent Using the `run` Function**:
#    - Use the `run` function of the agent to test its ability to calculate powers.
#    - Pass natural language queries to the agent and observe its responses.
# 

# In[ ]:


#TODO
agent.run("Calculate 5 to the power of 2.")


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# agent.run("Calculate 5 to the power of 2.")
# ```
# 
# </details>
# 

# ## Authors
# 

# [Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)
# 

# [Kunal Makwana](https://author.skills.network/instructors/kunal_makwana)
# 

# Copyright © IBM Corporation. All rights reserved.
# 
