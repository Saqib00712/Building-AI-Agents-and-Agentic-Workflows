#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Use Natural Language to Create Charts and Graphs**
# ## **Build Your Own Data Visualization Agent**
# 

# Estimated time needed: **30** minutes
# 

# ## Overview
# 

# Imagine you are a data analyst or a data scientist of a marketing team at an e-commerce company. The company needs to understand customer purchasing behaviors over the last year to tailor their upcoming holiday campaigns. Traditionally, this would involve complex SQL queries, data wrangling in Python, and perhaps building visual dashboards to interpret the results including analyzing spreadsheets, creating charts, and maybe even some statistical analysis—tasks that require considerable time and expertise.
# 
# With the integration of Langchain and LLMs, you can simply ask, "Show me a visualization of monthly sales trends by product category," or "Generate a heatmap of customer activity by region." The system would use the `create_pandas_dataframe_agent` to process the CSV data, and then dynamically generate visualizations such as line graphs, bar charts, or heatmaps in response to these queries. This not only speeds up the data analysis process but also allows team members who may not be tech-savvy to engage directly with the data and make informed decisions quickly. This approach fosters a more collaborative environment and ensures that strategic decisions are backed by real-time data insights, visually represented for easy comprehension.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/V_7__WU_jHJ1lOpTeSLxTQ/chat%20with%20data.png" width="50%" alt="indexing"/>
# 

# In this lab, you will learn how to seamlessly integrate data visualization into your conversational data analysis using Langchain and LLMs. Starting with CSV file data, you will use the `create_pandas_dataframe_agent` to build an interactive agent that not only understands and responds to your queries but also translates data responses into visual formats. You will explore how to dynamically generate charts, graphs, and heatmaps directly in response to natural language questions. This capability will enable you to visualize trends, compare figures, and spot patterns immediately, making your data analysis workflow both efficient and visually engaging. By the end of this project, you will have the skills to create a data conversational agent that acts as both analyst and visualizer, bringing data to life through dialogue.
# 
# In this lab, you are going to use Llama 3 LLM hosted on the IBM watsonx.ai platform.
# 

# ---------
# 

# ## __Table of contents__
# 
# <ol>
#     <li><a href="#Overview">Overview</a></li>
#     <li><a href="#Objectives">Objectives</a></li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
#             <li><a href="#Importing-required-libraries">Importing required libraries</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Data-set">Data set</a>
#         <ol>
#             <li><a href="#Load-the-data-set">Load the data set</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Load-LLM">Load LLM</a>
#         <ol>
#             <li><a href="#Talk-to-your-data">Talk to your data</a></li>
#             <li><a href="#Plot-your-data-with-natural-language">Plot your data with natural language</a></li>
#         </ol>
#     </li>
# </ol>
# 
# <a href="#Exercises">Exercises</a>
# <ol>
#     <li><a href="#Exercise-1---Relationship-between-parental-education-level-and-student-grades">Exercise 1. Relationship between parental education level and student grades</a></li>
#     <li><a href="#Exercise-2---Impact-of-internet-access-at-home-on-grades">Exercise 2. Impact of internet access at home on grades</a></li>
#     <li><a href="#Exercise-3---Explore-LLM's-code">Exercise 3. Explore LLM's code</a></li>
# </ol>
# 

# ## Objectives
# 
# 
# After completing the project, you should be able to:
# 
# - **Use LangChain with large language models**: Understand and apply the Langchain framework in conjunction with LLMs to interact with and analyze data stored in CSV files through natural language processing.
# - **Create conversational data agents**: Build a conversational agent that can understand and respond to natural language queries about data, enabling users to ask questions directly and receive immediate answers.
# - **Implement data visualization through dialogue**: Integrate data visualization tools within your conversational agent, allowing you to request and generate visual data representations such as graphs, charts, and heatmaps dynamically based on your queries.
# - **Enhance decision-making process**: Develop the capability to derive actionable insights from data via interactive dialogues and visual outputs, thereby improving the decision-making process and making data analysis accessible to non-technical stakeholders.
# 

# ----
# 

# ## Setup
# 

# This project is based on Jupyter Notebook. If you're not familiar with it, here's a quick guide on how to run code within it:
# 
# A Jupyter Notebook consists of cells. To execute a code cell, click on the cell that you want to run and click the 'Run' button, as shown in the picture.
# 

# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IwbhiH3Wwv-VK-J4rioTAw/run.png" width="50%" alt="indexing"/>
# 

# For this lab, you will be using the following libraries:
# 
# *   [`ibm-watson-ai`](https://ibm.github.io/watson-machine-learning-sdk/index.html) for using LLMs from IBM's watsonx.ai.
# *   [`LangChain`, `langchain-ibm`, `langchain-experimental`](https://www.langchain.com/) for using its agent function to interact with data.
# *   [`matplotlib`](https://matplotlib.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for additional plotting tools.
# *   [`seaborn`](https://seaborn.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0187ENSkillsNetwork31430127-2021-01-01) for visualizing the data.
# 

# ### Installing required libraries
# 
# The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:
# 
# **Note:** The version has been pinned here. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.
# 
# This might take approximately 1-2 minutes. 
# 
# As you use `%%capture` to capture the installation, you won't see the output process. But after the installation completes, you will see a number beside the cell.
# 

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install --user "ibm-watsonx-ai==0.2.6"\n!pip install --user "langchain==0.1.16" \n!pip install --user "langchain-ibm==0.1.4"\n!pip install --user "langchain-experimental==0.0.57"\n!pip install --user "matplotlib==3.8.4"\n!pip install --user "seaborn==0.13.2"\n')


# After you installat the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/build-a-hotdog-not-hotdog-classifier-guided-project/images/Restarting_the_Kernel.png" width="50%" alt="Restart kernel">
# 

# ### Importing required libraries
# 
# _It is recommended that you import all required libraries in one place (here):_
# 

# In[1]:


# You can use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

import matplotlib.pyplot as plt
import pandas as pd


# ## Dataset
# 

# In this lab, you will work on the Student Alcohol Consumption data set `student-mat.csv` by UCI Machine Learning as an example. For more information, see [Kaggle](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption). It is based on data collected from two secondary schools in Portugal. The students included in the survey were in mathematics and Portuguese courses.
# 

# The dataset you are using is for the mathematics course. The number of mathematics students involved in the collection was 395. The data collected in locations such as Gabriel Pereira and Mousinho da Silveira includes several pertinent values. Examples of such data are records of demographic information, grades, and alcohol consumption.
# 

# | Field     | Description                                                                 |
# |-----------|-----------------------------------------------------------------------------|
# | school    | GP/MS for the student's school                                              |
# | sex       | M/F for gender                                                              |
# | age       | 15-22 for the student's age                                                 |
# | address   | U/R for urban or rural, respectively                                        |
# | famsize   | LE3/GT3 for less than or greater than three family members                  |
# | Pstatus   | T/A for living together or apart from parents, respectively                 |
# | Medu      | 0 (none) / 1 (primary-4th grade) / 2 (5th - 9th grade) / 3 (secondary) / 4 (higher) for mother's education |
# | Fedu      | 0 (none) / 1 (primary-4th grade) / 2 (5th - 9th grade) / 3 (secondary) / 4 (higher) for father's education |
# | Mjob      | 'teacher,' 'health' care related, civil 'services,' 'at_home' or 'other' for the student's mother's job |
# | Fjob      | 'teacher,' 'health' care related, civil 'services,' 'at_home' or 'other' for the student's father's job |
# | reason    | reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other') |
# | guardian  | mother/father/other as the student's guardian                               |
# | traveltime| 1 (<15mins) / 2 (15 - 30 mins) / 3 (30 mins - 1 hr) / 4 (>1hr) for a time from home to school |
# | studytime | 1 (<2hrs) / 2 (2 - 5hrs) / 3 (5 - 10hrs) / 4 (>10hrs) for weekly study time |
# | failures  | 1-3/4 for the number of class failures (if more than three, then record 4)  |
# | schoolsup | yes/no for extra educational support                                        |
# | famsup    | yes/no for family educational support                                       |
# | paid      | yes/no for extra paid classes for Math or Portuguese                        |
# | activities| yes/no for extra-curricular activities                                      |
# | nursery   | yes/no for whether attended nursery school                                  |
# | higher    | yes/no for the desire to continue studies                                   |
# | internet  | yes/no for internet access at home                                          |
# | romantic  | yes/no for relationship status                                              |
# | famrel    | 1-5 scale on quality of family relationships                                |
# | freetime  | 1-5 scale on how much free time after school             |
# | goout     | 1-5 scale on how much student goes out with friends      |
# | Dalc      | 1-5 scale on how much alcohol consumed on weekdays       |
# | Walc      | 1-5 scale on how much alcohol consumed on the weekend    |
# | health    | 1-5 scale on health condition                            |
# | absences  | 0-93 number of absences from school                      |
# | G1        | 0-20 for the first-period grade                          |
# | G2        | 0-20 for the second-period grade                         |
# | G3        | 0-20 for the final grade                                 |
# 

# ### Load the data set
# 

# Execute the code in the following cell to load your dataset. This code reads the CSV file into a pandas DataFrame, making the data accessible for processing in Python.
# 

# In[2]:


df = pd.read_csv(
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZNoKMJ9rssJn-QbJ49kOzA/student-mat.csv"
)


# Let's examine the first five rows of the dataset to get a glimpse of the data structure and its contents.
# 

# In[3]:


df.head(5)


# You can also review the detailed information for each column in the dataset, focusing on the presence of null values and the specific data types of each column.
# 

# In[4]:


df.info()


# ## Load LLM
# 

# Execute the code in the cell below to load the llama-3-405b LLM model from watsonx.ai. 
# 
# Additionally, you will configure the LLM to interact with data by integrating it with Langchain's `create_pandas_dataframe_agent`.
# 

# In[16]:


# Create a dictionary to store credential information
credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}

# Indicate the model you would like to initialize. In this case, Llama 3 405B.
model_id    =  "ibm/granite-4-h-small"

# Initialize some watsonx.ai model parameters
params = {
        GenParams.MAX_NEW_TOKENS: 256, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }
project_id  = "skills-network" # <--- NOTE: specify "skills-network" as your project_id
space_id    = None
verify      = False

# Launch a watsonx.ai model
model = Model(
    model_id=model_id, 
    credentials=credentials, 
    params=params, 
    project_id=project_id, 
    space_id=space_id, 
    verify=verify
)

# Integrate the watsonx.ai model with the langchain framework
llm = WatsonxLLM(model = model)

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    return_intermediate_steps=True,  # set return_intermediate_steps=True so that model could return code that it comes up with to generate the chart
    handle_parsing_errors=True
)


# ## API Disclaimer
# This lab uses LLMs provided by **Watsonx.ai**. This environment has been configured to allow LLM use without API keys so you can prompt them for **free (with limitations)**. With that in mind, if you wish to run this notebook **locally outside** of Skills Network's JupyterLab environment, you will have to **configure your own API keys**. Please note that using your own API keys means that you will incur personal charges.
# 
# ### Running Locally
# If you are running this lab locally, you will need to configure your own API keys. This lab uses the `WatsonxLLM` module from `langchain`. The local configuration is shown below with instructions. **Replace all instances** of both modules with the completed modules below throughout the lab. **DO NOT** run the cell below if you aren't running locally, it will causes errors.
# 

# ### Interact with your data
# 

# Let's start with a simple interaction.
# 
# Ask LLM how many rows of data are in the CSV file.
# 

# In[17]:


response = agent.invoke("how many rows of data are in this file?")


# In[18]:


response['output']


# From the output above, the model reports that there are 395 rows of data in the file.
# 

# Let's verify this count using Python code to ensure accuracy.
# 

# In[19]:


len(df)


# The row count matches and is correct! 
# 

# Curious about the code the LLM generated and used to create this result?
# 
# Run the code in the cell below to reveal the underlying commands.
# 

# In[20]:


response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n')


# Surprisingly, the LLM uses the same code as you do.
# 

# Also, you could let LLM return some data that you are looking for based on the CSV file.
# 

# In[21]:


response = agent.invoke("Give me all the data where student's age is over 18 years old.")


# In[22]:


print(response)


# Let's get the code LLM used for charting this plot.
# 

# In[24]:


response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n')


# ### Plot your data with natural language
# 

# #### Task 1
# Generating a first visual on the data set to know the total number of male and female students in the data set.
# 
# You just need to tell the agent that "Plot the gender count with bars."
# 

# In[25]:


response = agent.invoke("Generate a bar chart to plot the gender count.")


# Let's see what code the LLM generated for ploting this chart.
# 

# In[26]:


print(response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))


# #### Task 2
# 
# Generating a pie chart to display the average value of weekend alcohol for each gender in the dataset.
# 
# You will use the prompt "Generate a pie chart to display the average value of Walc for each gender."
# 
# You may notice that the model generates two charts. The charts indicate the progressive improvement of the agent's code as it searches for the best way to answer your prompt, which improves the response to your query.
# 

# In[27]:


response = agent.invoke("Generate a pie chart to display average value of Walc for each Gender.")


# Let's get the code LLM used for charting this plot.
# 

# In[28]:


print(response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))


# #### Task 3
# 
# You can explore the impact of free time on grades based on the data.
# 

# In[29]:


response = agent.invoke("Create box plots to analyze the relationship between 'freetime' (amount of free time) and 'G3' (final grade) across different levels of free time.")


# Execute the code below to retrieve the Python script the LLM used for plotting.
# 

# In[30]:


print(response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))


# #### Task 4
# 
# You can explore the effect of alcohol consumption on academic performance.
# 

# In[31]:


response = agent.invoke("Generate scatter plots to examine the correlation between 'Dalc' (daily alcohol consumption) and 'G3', and between 'Walc' (weekend alcohol consumption) and 'G3'.")


# Execute the code below to retrieve the Python script the LLM used for plotting.
# 

# In[32]:


print(response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))


# # Exercises
# 

# ### Exercise 1 - Relationship between parental education level and student grades
# 

# In[33]:


# your code here
response = agent.invoke(
    "Generate scatter plots showing the relationship between "
    "'Medu' (mother's education level) and 'G3' (final grade), "
    "and between 'Fedu' (father's education level) and 'G3'. "
    
)


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# 
# response = agent.invoke(
#     "Generate scatter plots showing the relationship between "
#     "'Medu' (mother's education level) and 'G3' (final grade), "
#     "and between 'Fedu' (father's education level) and 'G3'. "
#     
# )
# 
# ```
# 
# </details>
# 

# ### Exercise 2 - Impact of internet access at home on grades
# 

# In[34]:


# your code here

response = agent.invoke("Use bar plots to compare the average final grades ('G3') of students with internet access at home versus those without ('internet' column).")


# <details>
#     <summary>Click here for a solution</summary>
#     
# ```python
# 
# response = agent.invoke("Use bar plots to compare the average final grades ('G3') of students with internet access at home versus those without ('internet' column).")
# 
# ```
# 
# </details>
# 

# ### Exercise 3 - Explore LLM's code
# 

# In[36]:


response = agent.invoke("Plot a scatter plot showing the correlation between the number of absences ('absences') and final grades ('G3') of students.")

for i in range(len(response['intermediate_steps'])):
    print(response['intermediate_steps'][i][0].tool_input.replace(';', '\n'))


# Can you find what code the model used to generate the plot for exploring the relationship between absences and academic performance?
# 
# You could run the corresponding code and from the response chain, you could see the code used from charting.
# 

# In[ ]:





# <details>
#     <summary>Click here for a solution</summary>
#     
# ```python
# 
# response = agent.invoke("Plot a scatter plot showing the correlation between the number of absences ('absences') and final grades ('G3') of students.")
# 
# for i in range(len(response['intermediate_steps'])):
#     print(response['intermediate_steps'][i][0].tool_input.replace(';', '\n'))
# 
# ```
# 
# </details>
# 

# ## Authors
# 

# [Kang Wang](https://author.skills.network/instructors/kang_wang)
# 
# Kang Wang is a Data Scientist in IBM. He is also a PhD Candidate in the University of Waterloo.
# 

# [Wojciech Fulmyk](https://author.skills.network/instructors/wojciech_fulmyk) <br>
# Wojciech "Victor" Fulmyk is a Data Scientist at IBM. He is also a PhD Candidate in Economics in the University of Calgary.
# 

# ## Other contributors
# 

# [Ricky Shi](https://author.skills.network/instructors/ricky_shi) <br>
# Ricky Shi is a data scientist at the Ecosystems Skills Network at IBM.
# 

# <!--## Change Log--!>
# 

# <!--|Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2024-05-10|0.2|Kang Wang & Wojciech Fulmyk|Initial version created|
# |2024-02-23|0.1|Elio Di Nino|Update library documentation|--!>
# 
# 

# ## Copyright © IBM Corporation. All rights reserved.
# 
