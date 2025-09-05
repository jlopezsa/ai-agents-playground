<div align="center" class="text-center">
<img src="/images/ai_agents_playground.png" alt="AI Agents Playground Banner" width="100%">

<!--h1>AI-AGENTS-PLAYGROUND</h1-->
<p><em>Explorando y aprendiendo con agentes inteligentes. ¡Este repositorio es un espacio para ir descubriendo las características de la IA!</em></p>

<img alt="last-commit" src="https://img.shields.io/github/last-commit/jlopezsa/ai-agents-playground?style=flat&amp;logo=git&amp;logoColor=white&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<img alt="repo-top-language" src="https://img.shields.io/github/languages/top/jlopezsa/ai-agents-playground?style=flat&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<img alt="repo-language-count" src="https://img.shields.io/github/languages/count/jlopezsa/ai-agents-playground?style=flat&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<p><em>Este proyecto fue creado utilizando las herramientas y tecnologías que se mencionan a continuación.:</em></p>

<img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-000000.svg?style=flat&logo=python&logoColor=white">

<img alt="LangChain" src="https://img.shields.io/badge/LangChain-2C9F75.svg?style=flat&logo=python&logoColor=white">

<img alt="Python" src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&amp;logo=Python&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<img alt="Poetry" src="https://img.shields.io/badge/Poetry-60A5FA.svg?style=flat&amp;logo=Poetry&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&amp;logo=Streamlit&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<img alt="TOML" src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat&amp;logo=TOML&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<img alt="Markdown" src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&amp;logo=Markdown&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">
</div>

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
