'''
Sample code to run a LangChain agent with Hugging Face and OpenAI models, using Gradio for the interface.
This code is designed to run an agent that can answer questions using various tools, including Wikipedia

Files List:
app_agents.py (Main application file to run the agent)
tools_script.py (Script defining the tools used by the agent)
requirements.txt (Dependencies for the application)
.env (Environment variables for API keys and model names)

Created by: Bhanu Chander V
Date: 15th June 2025
Modified: 20th June 2025
'''

## Import Necessary Libraries
import os 
import gradio as gr
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from tools_script import TOOLS_LIST, TOOLS_DESCRIPTIONS
from dotenv import load_dotenv


## # Load environment variables from .env file
load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')
OPENAI_MODEL = os.getenv ('OPENAI_MODEL')

## --
# --- System Prompt for the Agent ---

system_prompt = f"""
You are a general AI assistant, who can answer about general knowledge, historical facts, and also can analyze audios, images, and videos. You should think through the input question step-by-step and use tools if needed.

Use this reasoning format repeatedly:
Thought: (what you think is happening or what you want to do next)
Action: (the tool to use, if needed)
Action Input: (input to the tool)
Observation: (result of the tool call)

Repeat this process as needed. ONLY AFTER finishing your reasoning and/or tool use provide your final answer. 

You also have access to a set of tools, which you can use to answer the question. The available tools are:
{TOOLS_DESCRIPTIONS}

If the question is related to sports, awards, historical facts or similar topic that can be answered from wikipedia, you should use the 'wikipedia_and_generalknowledge_search'. 
If the question is about current events or news or similar current affairs category, you can utilize the tool 'current_events_news_search_tool' to fetch relevant page information and answer from it. 
If the tool returns a long text, table, or list, extract only the most relevant information/paragraphs or data from which you can derive the answer, and return that as your final answer.
You must not use multiple tools in a single call. Don't hallucinate.


**Examples of your Thought Process (ReAct):**
Q: Which country had the least number of athletes at the 1928 Summer Olympics?
Thought: I should look up historical Olympic data.
Action: wikipedia_and_generalknowledge_search
Action Input: least number of athletes 1928 Summer Olympics
Observation: The country with the least number of athletes was Luxembourg with 2 athletes.

Q: What are the top 3 programming languages?
Thought: This is common knowledge. I can answer msyelf

If even after 4 iterations, a tool usage is not useful then try to answer directly based on your knowledge without any hallucination. If you cannot answer then just say "no_answer" as YOUR FINAL ANSWER. 
"""

## --- Initialize the OpenAI chat model
chat_llm = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    model_name=OPENAI_MODEL,
    temperature=0.15,
    # max_tokens=10
)

# Initialize the agent with the tools and system prompt
agent = initialize_agent(
    tools=TOOLS_LIST,
    # llm=llm,
    llm=chat_llm,
    agent=AgentType.OPENAI_FUNCTIONS,#AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"system_message": system_prompt},
    max_iterations=10, # Increase as needed
    max_execution_time=1000, # Increase as needed
    early_stopping_method="generate",
    handle_parsing_errors=True,
    verbose=True,
    # return_intermediate_steps=True,
)


######....Sample Gradio Interface Code, without HITL (Human in the Loop) feedback.....#####
## --- Run Agent ---
def run_agent(query: str) -> str:
    """Pass the user prompt to the LangChain agent and return its answer."""
    return agent.run(query)

# --- Gradio Interface ---
demo = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask your question ....",
        label="Input your query below"
    ),
    outputs=gr.Textbox(label="Agent response"),
    title="LLM Agent Playground",
    description="Type a question, press Submit, and watch the agent answer.",
)


## --- Launch the Gradio app
if __name__ == "__main__":
    demo.launch()