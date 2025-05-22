import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["ORCHESTATOR_BASE_URL"]
model_name = os.environ["ORCHESTATOR_MODEL"]

memory = MemorySaver()

model = ChatOpenAI(
    base_url=base_url, 
    model=model_name
    )

def multiply(a: int, b:int) -> int:
    """Multiply a and b.
    Args:
        a: first int
        b: second int
    """
    return a*b

def add(a:int, b:int) -> int:
    """Adds a and b.
    Args:
        a: first int
        b: second int
    """
    return a+b

def divide(a: int, b:int) -> int:
    """Divide a and b.
    Args:
        a: first int
        b: second int
    """
    return a/b

tools = [add, multiply, divide]

llm_with_tools = model.bind_tools(tools, parallel_tool_calls=False)

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Add logic graph (edges)
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile the graph
react_graph_memory = builder.compile(checkpointer=memory)

# PAY ATTENTION HERE: see how we specify a thread
config = {"configurable": {"thread_id": "1"}}

# Enter the input
messages = [HumanMessage(content="Add 3 and 4.")]

# PAY ATTENTION HERE: see how we add config to referr to the thread_id
messages = react_graph_memory.invoke({"messages": messages},config)

for m in messages['messages']:
    m.pretty_print()

# PAY ATTENTION HERE: see how we check if the app has memory
messages = [HumanMessage(content="Multiply that by 2.")]

# Again, see how we use config here to referr to the thread_id
messages = react_graph_memory.invoke({"messages": messages}, config)

for m in messages['messages']:
    m.pretty_print()