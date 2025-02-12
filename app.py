# Standard library imports
import os
from typing import TypedDict, Union, List
import json

# Third-party imports
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize FastAPI app
app = FastAPI()

# Type definitions
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    next_agent: str

class Query(BaseModel):
    text: str

# Tools
@tool
def multiply(numbers: str) -> float:
    """Multiplies two space-separated numbers."""
    try:
        num1, num2 = map(float, numbers.split())
        return num1 * num2
    except:
        return "Error: Please enter two space-separated numbers"

# Initialize LLM models
analyzer_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
calculator_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Agent definitions
def analyzer(state: AgentState) -> AgentState:
    print("\n=== Analyzer Agent Started ===")
    print(f"Received message: {state['messages'][-1].content}")
    
    messages = state["messages"]
    response = analyzer_llm.invoke(
        [
            HumanMessage(content="""You are a helpful math assistant. Your task is to:
            1. Understand if the user's message is a mathematical operation
            2. If it's a multiplication operation, identify the numbers and indicate they need to be multiplied
            3. For any other type of mathematical operation or question, provide a helpful response
            
            Examples:
            For multiplication:
            - "5 times 3" -> needs_calculation: true, numbers: [5, 3]
            - "what is 6 * 7" -> needs_calculation: true, numbers: [6, 7]
            
            For other operations:
            - "2+42" -> needs_calculation: false, provide direct answer: "The sum of 2 and 42 is 44"
            - "what is 10/2" -> needs_calculation: false, provide direct answer: "10 divided by 2 is 5"
            
            Respond in JSON format:
            {
                "needs_calculation": true/false,
                "numbers": [number1, number2] if multiplication needed,
                "response": "direct answer for non-multiplication queries"
            }"""),
            *messages
        ]
    )
    
    # Parse the JSON response
    decision = json.loads(response.content)
    print(f"Analyzer decision: {decision}")
    
    if decision["needs_calculation"]:
        numbers = f"{decision['numbers'][0]} {decision['numbers'][1]}"
        state["messages"].append(AIMessage(content=f"CALCULATE: {numbers}"))
        state["next_agent"] = "calculator"
    else:
        state["messages"].append(AIMessage(content=decision["response"]))
        state["next_agent"] = "end"
    
    print(f"Next agent: {state['next_agent']}")
    print("=== Analyzer Agent Finished ===\n")
    return state

def calculator(state: AgentState) -> AgentState:
    print("\n=== Calculator Agent Started ===")
    last_message = state["messages"][-1].content
    print(f"Received calculation request: {last_message}")
    
    if last_message.startswith("CALCULATE:"):
        numbers = last_message.replace("CALCULATE:", "").strip()
        result = multiply.invoke(numbers)
        response = f"The multiplication result is: {result}"
    else:
        response = "No numbers received for multiplication"
    
    print(f"Calculation response: {response}")
    state["messages"].append(AIMessage(content=response))
    state["next_agent"] = "end"
    print("=== Calculator Agent Finished ===\n")
    return state

# Workflow configuration
def configure_workflow():
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("analyzer", analyzer)
    workflow.add_node("calculator", calculator)
    workflow.add_node("end", lambda x: x)

    # Set the entry point
    workflow.set_entry_point("analyzer")
    
    # Use add_conditional_edges for conditional routing
    workflow.add_conditional_edges(
        "analyzer",
        lambda x: x["next_agent"],
        {
            "calculator": "calculator",
            "end": "end"
        }
    )
    workflow.add_edge("calculator", "end")

    return workflow.compile()

# Initialize the workflow
graph = configure_workflow()

# API endpoints
@app.post("/ask")
async def ask_agents(query: Query):
    try:
        print("\n=== New Request ===")
        print(f"Query received: {query.text}")
        
        # Initialize the state
        state = AgentState(
            messages=[HumanMessage(content=query.text)],
            next_agent="analyzer"
        )
        
        # Execute the workflow
        result = graph.invoke(state)
        
        # Extract the last message as the response
        final_response = result["messages"][-1].content
        print(f"Final response: {final_response}")
        print("=== Request Completed ===\n")
        return {"response": final_response}
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {"error": str(e)}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
