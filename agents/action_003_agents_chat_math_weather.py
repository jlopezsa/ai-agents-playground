import os

import numexpr
import requests
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["ORCHESTATOR_BASE_URL"]
model_name = os.environ["ORCHESTATOR_MODEL"]

memory = MemorySaver()

model = ChatOpenAI(base_url=base_url, model=model_name)


class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built
    pass


# System message
sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


# Herramienta 1: Calculadora
def calculate(expression: str):
    """Evalúa una expresión matemática usando numexpr.

    Args:
        expression (str): Expresión matemática. Ej: "2 * (3 + 5)".

    Returns:
        str: Resultado como string o mensaje de error.

    Raises:
        SyntaxError: Si la expresión tiene formato inválido.
        TypeError: Si se usan tipos incorrectos.
    """
    print("--- Tool: calculate ---")
    try:
        return str(numexpr.evaluate(expression))
    except (SyntaxError, TypeError) as e:
        print(f"Error en cálculo: {e}")
        return "Error en el cálculo. Expresión no válida."
    except Exception as e:
        print(f"Error inesperado: {e}")
        return "Error interno en el cálculo."


# Herramienta 2: Informacion del clima
tools = [calculate]

llm_with_tools = model.bind_tools(tools, parallel_tool_calls=False)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def build_agent_graph():
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
    return react_graph_memory
