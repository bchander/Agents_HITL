'''
Sample code to run a LangChain agent with Hugging Face and OpenAI models, using Gradio for the interface.
This code is designed to run an agent that can answer questions using various tools, including Wikipedia.
I've also added Tracing and Monitoring of LLM applications using Langsmith and callbacks. This can be further enhanced

Files List:
app_agents_HITL.py (Main application file to run the agent)
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

##-----To enable trace in Langsmith, and if working with langchain, yo udon't need an additonal import for tracing, No extra code is needed to log a trace to LangSmith. Just run your LangChain code as you normally would.
# But before running the code, export the following environment variables:
# export LANGSMITH_API_KEY=your_api_key
# export LANGSMITH_TRACING=true

## # Load environment variables from .env file
load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')
LANGSMITH_API_KEY=os.getenv('LANGSMITH_API_KEY')
LANGSMITH_TRACING=os.getenv('LANGSMITH_TRACING')
LANGSMITH_ENDPOINT=os.getenv('LANGSMITH_ENDPOINT')

## --- System Prompt for the Agent ---

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
Thought: This is common knowledge.
Action: I can answer msyelf

If even after 4 iterations, a tool usage is not useful then try to answer directly based on your knowledge without any hallucination. If you cannot answer then just say "no_answer" as YOUR FINAL ANSWER. 
"""

## --- Initialize the OpenAI chat model
chat_llm = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    model_name=OPENAI_MODEL,
    temperature=0.15,
    # callbacks=[ls_handler, cost_handler],  # Add LangSmith and cost tracking handlers
    # max_tokens=10
)

# Initialize the agent with the tools and system prompt
agent = initialize_agent(
    tools=TOOLS_LIST,
    # llm=llm,
    llm=chat_llm,
    agent=AgentType.OPENAI_FUNCTIONS,#AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_kwargs={"system_message": system_prompt},
    max_iterations=6, # Increase as needed
    max_execution_time=1000, # Increase as needed
    early_stopping_method="generate",
    handle_parsing_errors=True,
    verbose=True,
    # callbacks=[ls_handler, cost_handler],  # Add LangSmith and cost tracking handlers
    # return_intermediate_steps=True,
)

# Using Langsmith callbacks, every LLM/tool event now streams to Langsmith (trace, tokens, latency etc.), and also adds tokens to cost_handler, so we can compute live costs


######....Sample Gradio Interface Code, with HITL (Human in the Loop) feedback.....#####
def run_agent(question, history):
    """
    Execute the agent and store raw output in `history` for later reference.
    """
    if not question.strip():
        return gr.update(value="⚠️ Please ask a question"), history

    answer = agent.run(question)
    # print(response.usage_metadata)
    history.append({"q": question, "a": answer}) # keep the pair in history
    return gr.update(value=answer), history


# --- Convert history into a Q & A list --
def history_to_context(history):
    """
    Convert the list of Q/A dicts into a plain-text block the LLM can read.
    Example format:
        Q1: ...
        A1: ...
        Q2: ...
        A2: ...
    """
    lines = []
    for turn in history:
        lines.append(f"Q: {turn['q']}\nA: {turn['a']}")
    return "\n\n".join(lines)


# --- Handle Feedback ---
def handle_feedback(choice, edited, last_answer, history):
    """
    • Yes  → accept as-is
    • Edit → ask the agent to verify the user's edit with full history
    • No   → ask the agent to try again with full history
    """

    context = history_to_context(history)   # all turns so far
    
    if choice == "Yes":
        final = last_answer.strip()
    elif choice == "Edit":
        verification_prompt = (
            f"{context}\n\n"
            "The human proposes this revised answer:\n"
            f"{edited}\n\n"
            "Verify whether the revised answer fully resolves the *original question*. "
            "Respond with either:\n"
            "APPROVED - if it is correct.\n"
            "CORRECTION - plus the correct answer.\n"
        )
        verified = agent.run(verification_prompt)
        final = verified.strip()
    else:  # "No"
        retry_prompt = (
            f"{context}\n\n"
            "The human has rejected the previous answer. "
            "Please provide a **new** answer to the original question. "
            "Only output the new final answer and don't include chain-of-thought."
        )
        new_answer = agent.run(retry_prompt)
        # Optionally add the retry into history
        history.append({"q": "(retry)", "a": new_answer})
        final = new_answer.strip()
    return final


## --- Gradio Interface Setup as blocks ---
with gr.Blocks(title="LLM Agent Playground (HITL)") as demo:
    gr.Markdown("## 🤖 Ask the agent and User can approve, reject, or edit agent's answer")

    #  Session state to keep the last output
    history_state = gr.State([])
    # cost_handler_state = gr.State(cost_handler)

    # --- Input row ---
    with gr.Row():
        user_box = gr.Textbox(
            lines=2, label="Input your query below", placeholder="Ask your question ..."
        )
        ask_btn = gr.Button("Run Agent")

    # --- Raw agent output ---
    agent_output = gr.Textbox(
        label="Agent response (may include Thought / Action etc.)", 
        interactive=False,
        # lines=10,
    )

    # --- Feedback widgets ---
    feedback_choice = gr.Radio(
        ["Yes", "No", "Edit"], label="Is the answer correct?", value="Yes"
    )
    edited_box = gr.Textbox(
        label="Edited answer (only used if you pick 'Edit')", lines=3
    )
    submit_btn = gr.Button("Submit Feedback")

    # --- Final answer (shown after feedback) ---
    final_answer_box = gr.Textbox(
        label="✅ Final (human-approved) answer", 
        lines=3, 
        interactive=False
    )

    # Connect buttons to functions
    ask_btn.click(
        fn=run_agent,
        inputs=[user_box, history_state],
        outputs=[agent_output, history_state],
        # postprocess=False,   # let us format cost below
    )

    submit_btn.click(
        fn=handle_feedback,
        inputs=[feedback_choice, edited_box, agent_output, history_state],
        outputs=[final_answer_box],
    )


## --- Launch the Gradio app
if __name__ == "__main__":
    demo.launch()