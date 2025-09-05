import streamlit as st
from agents.action_003_agents_chat_math_weather import build_agent_graph
from langchain_core.messages import HumanMessage


# --- Configuración de página ---
st.set_page_config(
    page_title="🤖 Chat con Agente IA (StateFull)",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)
    

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    st.markdown("## AI Agents Playground")
    st.markdown(
        """
        Bienvenido a tu asistente conversacional con agentes inteligentes.
        
        - Ejemplo de agente con memoria
        - Integra herramienta personalizada para realizar cálculos matemáticos simples. 
        - Basado en LangChain y Streamlit

        [Repositorio en GitHub](https://github.com/jlopezsa/ai-agents-playground/tree/main/projects/stateful_chat_agent)
        """
    )
    st.markdown("---")
    st.info("Desarrollado como proyecto personal para aprender sobre agentes IA.")


st.markdown(
    """
    <div style="text-align:center;">
        <!--img src="https://cdn.pixabay.com/photo/2017/01/31/13/14/robot-2027195_1280.png" width="200"/-->
        <h1 style="color:#0080ff;">🤖 Chat con Agente IA</h1>
        <p style="font-size:18px;">Interactúa con un agente inteligente capaz de resolver preguntas, cálculos y consultar el clima.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Init the agent
react_graph = build_agent_graph()

# Init chat session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Escribe tu mensaje")

config = {"configurable": {"thread_id": "1"}}

if not st.session_state.chat_history:
    st.info("💡 Ejemplo: ¿Cuál es la raíz cuadrada de 144?")
    
if st.button("🧹 Limpiar chat"):
    st.session_state.chat_history = []
    st.rerun()
    
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("El agente está pensando..."):
        messages = [HumanMessage(content=user_input)]
        result = react_graph.invoke({"messages": messages}, config)
        assistant_response = result["messages"][-1].content
    st.session_state.chat_history.append(("assistant", assistant_response))
    
# --- Historial de chat con estilos ---
for role, content in st.session_state.chat_history:
    if role == "user":
        avatar = "🧑‍🦰"
        bubble_color = "#06354a"
        align = "flex-end"
    else:
        avatar = "🤖"
        bubble_color = "#1E1E2E"
        align = "flex-start"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: {align}; margin-bottom: 8px;">
            <div style="background: {bubble_color}; padding: 12px 18px; border-radius: 16px; max-width: 80%; box-shadow: 0 2px 8px #0001;">
                <span style="font-size: 22px;">{avatar}</span> <span style="font-size: 16px;">{content}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )