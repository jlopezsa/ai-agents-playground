import streamlit as st
from agents.action_002_agents_edge_memory_tools import build_agent_graph
from langchain_core.messages import HumanMessage

# Init the agent
react_graph = build_agent_graph()

# Init chat session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🤖 Chat con Agente IA")

# User input
user_input = st.chat_input("Escribe tu mensaje")

config = {"configurable": {"thread_id": "1"}}

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    print("Invocando el agente ...")
    # Invoke the agent
    messages = [HumanMessage(content=user_input)]
    result = react_graph.invoke({"messages": messages},config)
    print("... resuesta exitosa.")
    print("Obteniendo respuesta ...")
    # Get response
    assistant_response = result["messages"][-1].content
    print("... respuesta obtenida con exito.")
    st.session_state.chat_history.append(("assistant", assistant_response))

# History chat
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)