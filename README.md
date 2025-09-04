# 🤖 AI Agents Playground

Bienvenido a **AI Agents Playground**, un entorno experimental para el desarrollo y prueba de agentes inteligentes basados en LangChain, LangGraph y OpenAI. Este repositorio está diseñado para facilitar la creación, orquestación y evaluación de agentes conversacionales y herramientas de automatización en distintos dominios.

## ✨ Características principales

- 🧠 Ejemplos de agentes con y sin memoria.
- 🛠️ Integración de herramientas externas (calculadora, búsqueda web, Wikipedia, etc.).
- 🔗 Orquestación de flujos conversacionales mediante grafos de estado.
- 📁 Proyectos de automatización e investigación en la carpeta [`projects`](projects/).

## 📂 Estructura del repositorio

- `agents/`: 🤖 Implementaciones de agentes y herramientas.
- `agents_in_action/`: 🧪 Ejemplos prácticos y pruebas de agentes.
- `projects/`: 📁 Proyectos de automatización e investigación.
- `logger_config.py`: 📝 Configuración de logging con color.
- `README.md`: 📄 Documentación principal.

## 🚀 Proyectos

| Proyecto                       | Descripción                                                                                                                                                            | Link                                                                                |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Stateless Chat Agent           | Un agente conversacional que no recuerda nada del pasado. Cada mensaje se trata de forma independiente, útil para respuestas rápidas y sin contexto                    | [projects/stateless_chat_agent](projects/stateless_chat_agent/)                     |
| Stateful chat agent            | Un agente conversacional que recuerda el historial de la conversación. Ideal para mantener el contexto entre mensajes y ofrecer respuestas más coherentes y naturales. | [projects/stateful_chat_agent](projects/stateful_chat_agent/)                       |
| Research Automation Multiagent | Automatización de entrevistas y generación de reportes usando agentes multi-rol.                                                                                       | [projects/research_automation_multiagent](projects/research_automation_multiagent/) |

➕ Próximamente nuevos proyectos ...

---

## 🛠️ Requisitos

- 🐍 Python 3.13+
- 📦 [Poetry](https://python-poetry.org/) para la gestión de dependencias

## ⚡ Instalación

```sh
poetry install
```

## 🏃 Uso

1. ⚙️ Configura tus variables de entorno en `.env` (ver `.env.example`).
2. ▶️ Ejecuta la app principal:
   ```sh
   streamlit run app.py
   ```
3. 🔍 Explora los proyectos en la carpeta [`projects`](projects/).

---

## 🤝 Contribución

Si deseas agregar nuevos agentes o proyectos, crea una nueva carpeta dentro de `projects/` y actualiza la tabla de arriba.

---

## 📜 Licencia
