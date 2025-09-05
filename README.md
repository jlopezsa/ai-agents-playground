<div align="center" class="text-center">
<img src="/images/ai_agents_playground_v2.png " alt="AI Agents Playground Banner" width="100%">

<!--h1>AI-AGENTS-PLAYGROUND</h1-->
<p><em>Explorando y aprendiendo con agentes inteligentes. ¡Este repositorio es un espacio para ir descubriendo las características de la IA!</em></p>

<img alt="last-commit" src="https://img.shields.io/github/last-commit/jlopezsa/ai-agents-playground?style=flat&amp;logo=git&amp;logoColor=white&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<img alt="repo-top-language" src="https://img.shields.io/github/languages/top/jlopezsa/ai-agents-playground?style=flat&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<img alt="repo-language-count" src="https://img.shields.io/github/languages/count/jlopezsa/ai-agents-playground?style=flat&amp;color=0080ff" class="inline-block mx-1" style="margin: 0px 2px;">
<p><em>Proyectos creados utilizando las herramientas y tecnologías, entre otras, que se mencionan a continuación:</em></p>

<img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-000000.svg?style=flat&logo=python&logoColor=white">

<img alt="LangChain" src="https://img.shields.io/badge/LangChain-2C9F75.svg?style=flat&logo=python&logoColor=white">

<img alt="Python" src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&amp;logo=Python&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<img alt="Poetry" src="https://img.shields.io/badge/Poetry-60A5FA.svg?style=flat&amp;logo=Poetry&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&amp;logo=Streamlit&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<img alt="TOML" src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat&amp;logo=TOML&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<img alt="Markdown" src="https://img.shields.io/badge/Markdown-000000.svg?style=flat&amp;logo=Markdown&amp;logoColor=white" class="inline-block mx-1" style="margin: 0px 2px;">

<div align="center" class="text-center"><h4>... en construcción</h4></div>
</div>

---

Bienvenido a **AI Agents Playground**, un entorno experimental para el desarrollo y prueba de agentes inteligentes basados en LangChain, LangGraph y OpenAI. El repositorio está diseñado para facilitar la creación, orquestación y evaluación de agentes conversacionales y herramientas de automatización en distintos dominios.

Este repositorio es un espacio para practicar y reproducir los ejemplos presentados en:

- Bootcamp: [Bootcamp 2025: Comprender y Crear Agentes IA Profesionales, De cero a nivel profesional: CrewAI, LangGraph, Multi-Agentes, Flows, etc.](https://www.udemy.com/course/bootcamp-2025-comprender-y-crear-agentes-ia-profesionales/?couponCode=KEEPLEARNING) creado por Julio Colomer
- Libro [Generative AI with LangChain](https://www.oreilly.com/library/view/generative-ai-with/9781837022014/) de Ben Auffarth y Leonid Kuligin

Además se encontrarán proyectos personales relacionados al tema.

## Contenido:

- [Contenido:](#contenido)
- [✨ Características principales](#-características-principales)
- [📂 Estructura del repositorio](#-estructura-del-repositorio)
- [🚀 Proyectos](#-proyectos)
- [🛠️ Requisitos](#️-requisitos)
- [⚡ Instalación](#-instalación)
- [🏃 Uso](#-uso)

## ✨ Características principales

🧠 **Agentes con y sin memoria, flujos RAG y ReACT:**  
 Se encuentran ejemplos prácticos de agentes que pueden recordar el historial de la conversación (stateful) o funcionar sin memoria (stateless), así como flujos avanzados como RAG (Retrieval-Augmented Generation) y ReACT (Reason + Act).

🛠️ **Integración de herramientas externas:**  
 Se aprende cómo conectar agentes con utilidades como búsqueda web, Wikipedia, Tavily, entre otras APIs, ampliando sus capacidades para resolver tareas más complejas.

🔗 **Orquestación de flujos conversacionales mediante grafos de estado:**  
 Se descubre cómo diseñar y controlar conversaciones complejas utilizando grafos de estado, permitiendo que los agentes sigan rutas lógicas y colaboren entre sí.

📁 **Proyectos de automatización e investigación:**  
 Proyectos completos y casos de uso reales se encuentran en la carpeta [`projects`](projects/), donde se aplican los conceptos aprendidos, del bootcamp y del libro, para resolver problemas prácticos y experimentar con nuevas ideas.

## 📂 Estructura del repositorio

A continuación se describe la organización de carpetas y archivos principales del repositorio. Cada sección agrupa ejemplos, utilidades y proyectos para facilitar la exploración y el aprendizaje sobre agentes inteligentes y sus aplicaciones.

- `agents/`: 🤖 Implementaciones de agentes y herramientas. Estos agentes se utilizan en proyectos.
- `agents_in_action/`: 🧪 Ejemplos prácticos y pruebas de agentes. En estos archivos se prueban y hacen pruebas de los agentes.
- `projects/`: 📁 Proyectos de automatización e investigación.
- `logger_config.py`: 📝 Configuración de logging con color.
- `README.md`: 📄 Documentación principal.

## 🚀 Proyectos

- [Stateless Chat Agent](projects/stateless_chat_agent): Un agente conversacional que no recuerda nada del pasado. Cada mensaje se trata de forma independiente, útil para respuestas rápidas y sin contexto.
- [Stateful chat agent](projects/stateful_chat_agent/): Un agente conversacional que recuerda el historial de la conversación. Ideal para mantener el contexto entre mensajes y ofrecer respuestas más coherentes y naturales.
- [Research Automation Multiagent](projects/research_automation_multiagent/): Automatización de entrevistas y generación de reportes usando agentes multi-rol.
- [Content Marketing Manager](https://github.com/jlopezsa/ai-content-marketing): (repositorio **ai-content-marketing**) Agentes especializados que colaboran para realizar investigaciones, redactar blogs optimizados y generar mensajes simples para ser publicadas en redes sociales.

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
