# Standard library imports
import os
from typing import TypedDict, Union, List
import json
import uuid
from datetime import datetime
import logging

# Third-party imports
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.graph import StateGraph
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
mongo_url = os.getenv("MONGO_DB_DATABASE_URL")  

# Type definitions
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    next_agent: str
    needs_approval: bool
    approved: bool
    request_id: str

class Query(BaseModel):
    text: str

class ApprovalRequest(BaseModel):
    approve: bool

class PendingRequest(BaseModel):
    _id: str
    timestamp: datetime
    state: dict
    calculation: str
    original_query: str

class DatabaseConnection:
    def __init__(self):
        self.client = AsyncIOMotorClient(mongo_url)
        self.db = self.client['mongo-db']
        
    async def initialize(self):
        try:
            # Test the connection
            await self.client.admin.command('ping')
            
            # Ensure the collection exists
            if 'pending_requests' not in await self.db.list_collection_names():
                await self.db.create_collection('pending_requests')
            print("Database connection initialized")
        except Exception as e:
            print(f"Error initializing database connection: {e}")
            raise e
    
    def get_db(self):
        return self.db
    
    async def close_db(self):
        await self.client.close()

db_connection = DatabaseConnection()

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)

# Agent definitions
def analyzer(state: AgentState) -> AgentState:
    logger.info(f"Starting analysis for request [{state['request_id']}]")
    logger.debug(f"Input message: '{state['messages'][-1].content}'")
    
    messages = state["messages"]
    response = llm.invoke(
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
    logger.debug(f"Analyzer decision: {decision}")
    
    if decision["needs_calculation"]:
        numbers = f"{decision['numbers'][0]} {decision['numbers'][1]}"
        state["messages"].append(AIMessage(content=f"CALCULATE: {numbers}"))
        state["next_agent"] = "approver"
        state["needs_approval"] = True
        
        pending_request = {
            "_id": state["request_id"],
            "timestamp": datetime.now(),
            "state": state,
            "calculation": numbers,
            "original_query": state["messages"][0].content
        }
        state["pending_data"] = pending_request
        logger.info(f"Request [{state['request_id']}] requires approval for calculation: {numbers}")
    else:
        state["messages"].append(AIMessage(content=decision["response"]))
        state["next_agent"] = "end"
        state["needs_approval"] = False
        logger.info(f"Request [{state['request_id']}] processed directly with response")
    
    logger.debug(f"Analysis complete. Next agent: {state['next_agent']}")
    return state

def approver(state: AgentState) -> AgentState:
    logger.info(f"Approver processing request [{state['request_id']}]")
    
    if not state.get("approved", False):
        logger.info(f"Request [{state['request_id']}] queued for approval")
        state["next_agent"] = "wait_approval"
        return state
    
    logger.info(f"Request [{state['request_id']}] approved - proceeding to calculation")
    state["next_agent"] = "calculator"
    return state

def calculator(state: AgentState) -> AgentState:
    logger.info(f"Calculator processing request [{state['request_id']}]")
    last_message = state["messages"][-1].content
    logger.debug(f"Calculation input: '{last_message}'")
    
    if last_message.startswith("CALCULATE:"):
        numbers = last_message.replace("CALCULATE:", "").strip()
        result = multiply.invoke(numbers)
        response = f"The multiplication result is: {result}"
        logger.info(f"Request [{state['request_id']}] calculation complete: {result}")
    else:
        response = "Error: Invalid calculation format received"
        logger.error(f"Request [{state['request_id']}] received invalid calculation format")
    
    state["messages"].append(AIMessage(content=response))
    state["next_agent"] = "end"
    return state

# Workflow configuration
def configure_workflow():
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("analyzer", analyzer)
    workflow.add_node("approver", approver)
    workflow.add_node("calculator", calculator)
    workflow.add_node("wait_approval", lambda x: x)
    workflow.add_node("end", lambda x: x)

    # Set the entry point
    workflow.set_entry_point("analyzer")
    
    workflow.add_conditional_edges(
        "analyzer",
        lambda x: x["next_agent"],
        {
            "approver": "approver",
            "end": "end"
        }
    )
    
    workflow.add_conditional_edges(
        "approver",
        lambda x: x["next_agent"],
        {
            "calculator": "calculator",
            "wait_approval": "wait_approval"
        }
    )
    
    workflow.add_edge("calculator", "end")

    return workflow.compile()

# Initialize the workflow
graph = configure_workflow()

# Update the lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db_connection.initialize()
        app.state.db = db_connection.get_db()
        logger.info("Database connection established and collection created")
        yield
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        yield
    finally:
        await db_connection.close_db()
        logger.info("Database connection closed")

app = FastAPI(lifespan=lifespan)

# API endpoints
@app.post("/ask")
async def ask_agents(request: Request, query: Query):
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"New calculation request [{request_id}] received")
        logger.debug(f"Query: '{query.text}'")
        
        state = AgentState(
            messages=[HumanMessage(content=query.text)],
            next_agent="analyzer",
            needs_approval=False,
            approved=False,
            request_id=request_id
        )
        
        result = graph.invoke(state)
        
        if result["next_agent"] == "wait_approval":
            # Convert state to serializable dict
            state_dict = {
                "messages": [
                    {"type": msg.__class__.__name__, "content": msg.content} 
                    for msg in result["messages"]
                ],
                "next_agent": result["next_agent"],
                "needs_approval": True,
                "approved": False,
                "request_id": request_id
            }
            
            pending_request = {
                "_id": request_id,
                "timestamp": datetime.now(),
                "state": state_dict,
                "calculation": result["messages"][-1].content,
                "original_query": query.text
            }
            
            await request.app.state.db.pending_requests.insert_one(pending_request)
            logger.info(f"Request [{request_id}] stored in database pending approval")
            
            return {
                "status": "waiting_approval",
                "request_id": request_id,
                "message": "Calculation needs approval",
                "proposed_calculation": result["messages"][-1].content
            }
        
        final_response = result["messages"][-1].content
        logger.info(f"Request [{request_id}] completed successfully")
        return {"response": final_response}
    
    except Exception as e:
        logger.error(f"Error processing request [{request_id}]: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.post("/approve/{request_id}")
async def approve_calculation(request: Request, request_id: str, approval: ApprovalRequest):
    try:
        logger.info(f"Received approval request for ID: {request_id}")
        
        saved_request = await request.app.state.db.pending_requests.find_one({"_id": request_id})
        if not saved_request:
            logger.warning(f"Request ID {request_id} not found in MongoDB")
            return {"error": "Request ID not found"}
        
        # Reconstituim mesajele din dicționar
        messages = [
            HumanMessage(content=msg["content"]) if msg["type"] == "HumanMessage"
            else AIMessage(content=msg["content"])
            for msg in saved_request["state"]["messages"]
        ]
        
        state = AgentState(
            messages=messages,
            next_agent=saved_request["state"]["next_agent"],
            needs_approval=saved_request["state"]["needs_approval"],
            approved=approval.approve,
            request_id=request_id
        )
        
        result = graph.invoke(state)
        final_response = result["messages"][-1].content
        logger.info(f"Request {request_id} approved and processed successfully")
        
        return {"response": final_response}
    
    except Exception as e:
        logger.error(f"Error processing approval: {str(e)}", exc_info=True)
        return {"error": str(e)}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)