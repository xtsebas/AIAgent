import os
import datetime
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

load_dotenv()


def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {question}-->{answer}\n")


def load_history():
    if os.path.exists("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()
    return []


def display_history_as_conversation(history):
    st.markdown("## Historial de Conversación")
    if not history:
        st.markdown("No hay interacciones previas.")
    else:
        for entry in history:
            try:
                timestamp, interaction = entry.split(":", 1)
                question, answer = interaction.split("-->", 1)
                st.markdown(f"**{timestamp.strip()}**")
                st.markdown(f"**Pregunta:** {question.strip()}")
                st.markdown(f"**Respuesta:** {answer.strip()}")
                st.markdown("---")
            except ValueError:
                st.markdown(f"**Entrada no procesable:** {entry.strip()}")
                st.markdown("---")


def main():
    st.set_page_config(page_title="Interactive Python Agent", page_icon="🤖", layout="wide")

    st.title("🤖 Interactive Python Agent")
    st.markdown("### Descripción del Bot")
    st.markdown("""
    Este bot tiene la capacidad de ejecutar código Python para responder preguntas y también de responder preguntas basadas en datos de diferentes archivos CSV. 
    Utiliza el input para escribir preguntas específicas o selecciona una de las opciones para ejecutar ejemplos predefinidos.
    """)

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4o"),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # Definición de agentes para cada archivo CSV
    games_players = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/games_by_players.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    games_teams = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/games_by_teams.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    main_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/main.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    matches_players_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/matches_by_players.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    matches_teams_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/matches_by_teams.csv",
        verbose=True,
        allow_dangerous_code=True,
    )
    players_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="CSV/players_db.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    # Funciones de envoltura para cada agente CSV
    def games_players_wrapper(input_prompt: str) -> dict[str, Any]:
        return games_players.invoke({"input": input_prompt})

    def games_teams_wrapper(input_prompt: str) -> dict[str, Any]:
        return games_teams.invoke({"input": input_prompt})

    def main_wrapper(input_prompt: str) -> dict[str, Any]:
        return main_agent.invoke({"input": input_prompt})

    def matches_players_wrapper(input_prompt: str) -> dict[str, Any]:
        return matches_players_agent.invoke({"input": input_prompt})
    
    def matches_teams_wrapper(input_prompt: str) -> dict[str, Any]:
        return matches_teams_agent.invoke({"input": input_prompt})
    
    def players_wrapper(input_prompt: str) -> dict[str, Any]:
        return players_agent.invoke({"input": input_prompt})

    tools = [
        Tool(name="Python Agent", func=python_agent_executor_wrapper,
             description="""Useful when you need to transform natural language to python and execute the python code, 
             returning the results of the code execution DOES NOT ACCEPT CODE AS INPUT"""),
        Tool(name="Games by Players Agent", func=games_players_wrapper,
             description="Provides detailed answers about games played by players, including information about the "
                         "game, player, and the number of games played, using data from games_by_players.csv."),
        Tool(name="Games by Teams Agent", func=games_teams_wrapper,
             description="Provides detailed answers about games played by teams, including information about the game, "
                         "team, and the number of games played, using data from games_by_teams.csv."),
        Tool(name="Main Agent", func=main_wrapper,
             description="Provides detailed answers about the main information of the tournament, including the name, "
                         "location, and date of the tournament, using data from main.csv."),
        Tool(name="Matches by Players Agent", func=matches_players_wrapper,
             description="Provides detailed answers about matches played by players, including information about the "
                         "match, player, and the number of matches played, using data from matches_by_players.csv."),
        Tool(name="Matches by Teams Agent", func=matches_teams_wrapper,
             description="Provides detailed answers about matches played by teams, including information about the match, "
                         "team, and the number of matches played, using data from matches_by_teams.csv."),
        Tool(name="Players Agent", func=players_wrapper,
             description="Provides detailed answers about the players, including information about the player, team, "
                         "and the number of games played, using data from players_db.csv."),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools
    )

    grand_agent_executor = AgentExecutor(
        agent=grand_agent,
        tools=tools,
        verbose=True,
    )

    # Cargar historial como conversación
    history = load_history()
    display_history_as_conversation(history)

    # Sección de Python REPL
    st.markdown("## Python REPL")
    python_options = [
        "La división de los números 4000 sobre 40",
        "Crea un juego básico de snake con la librería pygame",
        "Genera un patrón piramidal de asteriscos de tamaño 5",
    ]
    python_example = st.selectbox("Ejemplos de Python", python_options, key="python_example")
    if st.button("Ejecutar Python", key="execute_python"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown("### Procesando tu respuesta...")

        try:
            response = python_agent_executor_wrapper(python_example)
            if 'output' in response:
                st.markdown("### Respuesta del agente:")
                st.code(response['output'], language="python")
                save_history(python_example, response['output'])
            else:
                st.error("No hubo salida del agente.")
        except Exception as e:
            st.error(f"Error al ejecutar el agente: {str(e)}")

    st.markdown("---")  # Separador

    csv_question = st.text_input("Escribe tu pregunta", key="csv_input")
    if st.button("Ejecutar pregunta", key="execute_query"):
        loading_placeholder = st.empty()
        loading_placeholder.markdown("### Procesando tu respuesta...")
        try:
            response = grand_agent_executor.invoke({"input": csv_question})
            if 'output' in response:
                st.markdown("### Respuesta del agente:")
                st.write(response['output'])
                save_history(csv_question, response['output'])
            else:
                st.error("No se pudo obtener una respuesta adecuada para la pregunta.")
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {str(e)}")


if __name__ == "__main__":
    main()