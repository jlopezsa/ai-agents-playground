import io
import os
from typing import List

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from PIL import Image
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

_ = load_dotenv(find_dotenv())

openai_api_key = os.environ["OPENAI_API_KEY"]
base_url = os.environ["ORCHESTATOR_BASE_URL"]
model_name = os.environ["ORCHESTATOR_MODEL"]


llm = ChatOpenAI(base_url=base_url, model=model_name)


# Clase Analyst
# Definir agentes con personalidades específicas
# Mantener consistencia en los roles de los agentes
# Generar descripciones automáticas para prompts
# Facilitar la serialización/deserialización de agentes
class Analyst(BaseModel):
    # Field se usa para validaciones y metadatos de los atributos de la clase
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(description="Name of the analyst.")
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )

    # @property es una característica nativa de Python que permite definir un método que se puede acceder como si fuera un atributo.
    @property
    def persona(self):
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


# Propósito: Contiene una lista de analistas y los describe como un grupo relacionado con un tema.
class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )


# Propósito: Almacena el estado del proceso de generación de analistas.
class GenerateAnalystsState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions


# Principal prompt
analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
{human_analyst_feedback}
3. Determine the most interesting themes based upon documents and / or feedback above.
4. Pick the top {max_analysts} themes.
5. Assign one analyst to each theme."""


# ========== Definiendo los nodos ===============
# Node: create_analysts
def create_analysts(state: GenerateAnalystsState):
    """Create analysts"""

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )

    # Generate question
    analysts = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate the set of analysts.")]
    )

    # Write the list of analysis to state
    return {"analysts": analysts.analysts}


# Node: human_feedback
def human_feedback(state: GenerateAnalystsState):
    """No-op node that should be interrupted on"""
    pass


# Conditional edge
def should_continue(state: GenerateAnalystsState):
    """Return the next node to execute"""

    # Check if human feedback
    human_analyst_feedback = state.get("human_analyst_feedback", None)

    if human_analyst_feedback:
        return "create_analysts"

    # Otherwise end
    return END


# ==================== Construyendo el grafo ===================
# Add nodes and edges
# State of graph
builder = StateGraph(GenerateAnalystsState)

# Nodes
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)

# Edges
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges(
    "human_feedback", should_continue, ["create_analysts", END]
)

# Memory
memory = MemorySaver()

# Compile
graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)


# ===================== Create a image graph =====================
graph_image = graph.get_graph(xray=True)
png_bytes = graph_image.draw_mermaid_png()
image = Image.open(io.BytesIO(png_bytes))
image.save("ai_analyst_generator.png")
print("✅ Graph image saved as graph.png")

# =================== Call Agent without interruptions =========================
max_analysts = 3
topic = "The main topic of use artificial inteligence in channel model wirelles communications"
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(
    {
        "topic": topic,
        "max_analysts": max_analysts,
    },
    thread,
    stream_mode="values",  # Permite procesar los resultados a medida que se generan, en lugar de esperar a que termine toda la ejecución.
):

    # Review
    analysts = event.get("analysts", "")

    if analysts:
        for analyst in analysts:
            print("-" * 25, "🤖", "-" * 25)
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")


# Get state and look at next node
state = graph.get_state(thread)
state.next
print("---> State next: ", state.next)

# =================== Call Agent with interruptions =========================
# Human Feedback simulation
# Actualiza el estado del graph como si el proceso se hubiera detenido para recibir retroalimentación humana y se hubiera recibido un input.
# as_node="human_feedback" – Indica que la retroalimentación debe aplicarse en el nodo de retroalimentación humana.
graph.update_state(
    thread,
    {
        "human_analyst_feedback": "Add in someone from a startup to add an entrepreneur perspective"
    },
    as_node="human_feedback",
)

print("====> Second execution <=====")

# Continue the graph execution
for event in graph.stream(None, thread, stream_mode="values"):
    # Review
    analysts = event.get("analysts", "")
    if analysts:
        for analyst in analysts:
            print(f"Name: {analyst.name}")
            print(f"Affiliation: {analyst.affiliation}")
            print(f"Role: {analyst.role}")
            print(f"Description: {analyst.description}")
            print("-" * 50)


# No more human feedback
# If we are satisfied, then we simply supply no feedback
further_feedack = None

graph.update_state(
    thread, {"human_analyst_feedback": further_feedack}, as_node="human_feedback"
)


# Continue the graph execution to end
for event in graph.stream(None, thread, stream_mode="updates"):
    print("--Node--")
    node_name = next(iter(event.keys()))
    print(node_name)

# Obtiene el state final del graph después de la ejecución.
final_state = graph.get_state(thread)
analysts = final_state.values.get("analysts")


print("========> FINAL ANALYSTS <===========")
for analyst in analysts:
    print("-" * 25, "🤖", "-" * 25)
    print(f"Name: {analyst.name}")
    print(f"Affiliation: {analyst.affiliation}")
    print(f"Role: {analyst.role}")
    print(f"Description: {analyst.description}")
