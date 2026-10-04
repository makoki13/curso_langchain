# ==============================================================================
# INSTRUCCIONES DE INSTALACIÓN (comentadas, para referencia)
# ==============================================================================

# Estas líneas muestran los paquetes necesarios para ejecutar el código.
# Si necesitas instalarlos, descomenta y ejecuta en tu terminal:
#pip install langchainhub
#pip install langchain langchain-community langchain-experimental duckduckgo-search wikipedia
#pip install langchainhub


# ==============================================================================
# IMPORTACIONES
# ==============================================================================

# Importamos 'uuid' porque LangGraph lo requiere internamente para inspeccionar
# los type hints de las herramientas. Aunque no lo usemos directamente en nuestro
# código, si no lo importamos, LangGraph lanza un NameError al crear el agente.
# El comentario '# noqa: F401' le dice al linter (Pylance/Pyright) que ignore
# la advertencia de "importación no utilizada".
import uuid  # ✅ AÑADIR ESTO - Requerido por PythonREPLTool  # noqa: F401

# Importamos 'datetime' para obtener la fecha actual del sistema.
# Esta fecha se inyectará en el prompt del sistema para que el LLM sepa
# en qué momento temporal se encuentra, evitando que use su conocimiento
# interno desactualizado para añadir años incorrectos a las búsquedas.
from datetime import datetime

# Importamos el módulo 'wikipedia' directamente.
# Este es el cliente Python oficial de la API de Wikipedia.
# Lo usaremos para configurar el User-Agent y pasarlo al wrapper de LangChain.
import wikipedia

# 'create_agent' es la función de LangGraph que crea un agente ReAct moderno.
# Reemplaza a la antigua combinación de 'create_react_agent' + 'AgentExecutor'.
# El agente devuelto es directamente ejecutable con .invoke().
from langchain.agents import create_agent

# 'DuckDuckGoSearchRun' es la herramienta de búsqueda en internet.
# Permite al agente realizar consultas en DuckDuckGo para obtener
# información actualizada que no está en su conocimiento interno.
from langchain_community.tools import DuckDuckGoSearchRun

# 'WikipediaAPIWrapper' es el wrapper de LangChain para la API de Wikipedia.
# Actúa como intermediario entre el agente y el módulo 'wikipedia' de Python,
# gestionando automáticamente las llamadas a la API y el formato de los resultados.
from langchain_community.utilities import WikipediaAPIWrapper

# 'HumanMessage' representa un mensaje del usuario en el sistema de mensajes
# de LangChain. Los agentes de LangGraph esperan recibir una lista de mensajes
# con roles específicos (human, ai, system, tool).
from langchain_core.messages import HumanMessage

# 'tool' es un decorador de LangChain que convierte una función Python normal
# en una herramienta que puede ser usada por un agente.
# El decorador extrae automáticamente:
# - El nombre de la herramienta (del nombre de la función)
# - La descripción (del docstring de la función)
# - Los parámetros esperados (de los type hints de la función)
from langchain_core.tools import tool

# 'PythonREPL' es la utilidad que permite ejecutar código Python en tiempo real.
# REPL significa "Read-Eval-Print Loop" (bucle de lectura-evaluación-impresión).
# El agente puede usar esta herramienta para ejecutar cálculos, procesar datos
# o resolver problemas matemáticos escribiendo código Python.
from langchain_experimental.utilities import PythonREPL

# 'ChatGroq' es el wrapper de LangChain para conectarse a la API de Groq.
# Groq ofrece inferencia ultrarrápida de modelos de lenguaje gracias a su
# hardware especializado (LPU - Language Processing Unit).
from langchain_groq import ChatGroq

# Configuramos el User-Agent para las peticiones a Wikipedia.
# Esto es CRÍTICO porque Wikipedia bloquea peticiones sin User-Agent apropiado.
# Sin esta línea, la herramienta de Wikipedia lanzaría un JSONDecodeError
# al recibir una respuesta vacía o de error de la API.
# El formato recomendado es: "NombreApp/Versión (contacto)"
wikipedia.set_user_agent("LangChainAgent/1.0 (curso-langchain)")


# ==============================================================================
# MODELO DE LENGUAJE (LLM)
# ==============================================================================

# Creamos la instancia del modelo de lenguaje usando ChatGroq.
# Este modelo será el "cerebro" del agente: decidirá qué herramientas usar,
# cómo interpretar los resultados y cómo formular la respuesta final.
llm = ChatGroq(
    # El modelo específico que queremos usar.
    # "qwen/qwen3.8-27b" es un modelo de 27 mil millones de parámetros
    # desarrollado por Alibaba. Es rápido, eficiente y gratuito en Groq.
    # El formato "proveedor/nombre-modelo" indica que Groq aloja este modelo.
    # NOTA: El modelo "openai/gpt-oss-safeguard-20b" está comentado porque
    # es un clasificador de seguridad, no un modelo generativo.
    model="qwen/qwen3.8-27b",

    # 'temperature' controla el nivel de aleatoriedad en las respuestas.
    # 0 = Totalmente determinista. Dada la misma entrada, siempre produce
    #     exactamente la misma salida. Ideal para tareas que requieren precisión
    #     y consistencia, como búsquedas de información factual.
    # Valores más altos (0.7, 1.0) harían las respuestas más creativas pero
    # menos predecibles.
    temperature=0,
)


# ==============================================================================
# PROMPT DEL SISTEMA (INSTRUCCIONES PARA EL AGENTE)
# ==============================================================================

# Definimos el prompt del sistema usando un f-string para poder inyectar
# la fecha actual dinámicamente. Este prompt establece las reglas que el
# agente debe seguir en todas sus interacciones.
#
# IMPORTANTE: Usamos f-string (f''') para que {datetime.now().strftime(...)}
# se evalúe en tiempo de ejecución y se convierta en una fecha real como
# "4 de octubre de 2026".
system_prompt = f'''
Eres un asistente útil que responde preguntas en italiano lo mejor que puedas.

FECHA ACTUAL: {datetime.now().strftime("%d de %B de %Y")}

REGLAS CRÍTICAS PARA BÚSQUEDAS:
1. Tu conocimiento interno está DESACTUALIZADO. No confíes en él para fechas, resultados deportivos, eventos recientes ni noticias.
2. Cuando busques en internet, NUNCA añadas años específicos a la búsqueda basándote en tu conocimiento interno.
3. Busca siempre de forma genérica. Por ejemplo:
   - ❌ MAL: "quién ganó la Copa del Mundo 2022"
   - ✅ BIEN: "último campeón Copa del Mundo FIFA"
   - ❌ MAL: "presidente de Francia 2023"
   - ✅ BIEN: "presidente actual de Francia"
4. SIEMPRE usa las herramientas para verificar información factual.
5. Si necesitas información sobre un tema, usa Wikipedia o DuckDuckGo.
6. Si necesitas hacer cálculos o procesar datos, usa Python REPL.
'''

# Explicación detallada del prompt:
# - "FECHA ACTUAL": Le dice al agente en qué año está, evitando que use su
#   conocimiento interno desactualizado (que puede ser de 2022, 2023 o 2024).
# - "REGLAS CRÍTICAS": Instrucciones explícitas para evitar el problema del
#   "sesgo del conocimiento interno", donde el LLM añade años incorrectos a
#   las búsquedas basándose en lo que "cree" que es la fecha actual.
# - Los ejemplos (❌ MAL / ✅ BIEN) son cruciales porque muestran al modelo
#   exactamente qué tipo de búsquedas debe hacer y cuáles debe evitar.


# ==============================================================================
# DEFINICIÓN DE LAS HERRAMIENTAS (TOOLS)
# ==============================================================================

# Creamos una instancia de PythonREPL.
# Este objeto permite ejecutar código Python en un entorno aislado.
# El agente lo usará para resolver cálculos matemáticos, procesar datos
# o ejecutar cualquier lógica programática necesaria.
python_repl = PythonREPL()

# Definimos la herramienta de Python REPL usando el decorador @tool.
# El decorador convierte esta función en una herramienta que el agente puede usar.
#
# Parámetros:
# - code: str -> El código Python que el agente quiere ejecutar
#
# Retorna:
# - str -> La salida del código ejecutado o un mensaje de error
#
# El docstring ("""...""") es MUY IMPORTANTE porque se convierte en la
# descripción de la herramienta. El agente lee esta descripción para decidir
# CUÁNDO usar esta herramienta.
@tool
def python_repl_tool(code: str) -> str:
    """Útil cuando necesitas usar Python para responder una pregunta. Debes ingresar código Python válido."""
    # Usamos try-except para capturar cualquier error de ejecución.
    # Si el código Python tiene un error de sintaxis o lanza una excepción,
    # en lugar de romper el agente, devolvemos un mensaje de error que el
    # agente puede leer y usar para corregir el código en el siguiente intento.
    try:
        return python_repl.run(code)
    except Exception as e:
        return f"Error ejecutando Python: {str(e)}"


# Definimos la herramienta de Wikipedia.
# Esta herramienta permite al agente buscar información enciclopédica.
#
# Parámetros:
# - query: str -> El tema, persona o concepto a buscar en Wikipedia
#
# Retorna:
# - str -> Un resumen del artículo encontrado o un mensaje de error
@tool
def wikipedia_tool(query: str) -> str:
    """Útil cuando necesitas buscar información enciclopédica sobre un tema, país o persona."""
    try:
        # Creamos el wrapper de Wikipedia pasándole el cliente configurado
        # con el User-Agent apropiado (sin esto, Wikipedia bloquearía la petición).
        api_wrapper = WikipediaAPIWrapper(wiki_client=wikipedia)
        # Ejecutamos la búsqueda y devolvemos el resultado.
        # El wrapper automáticamente:
        # - Busca el artículo más relevante
        # - Extrae un resumen (top_k_results)
        # - Lo formatea como texto legible
        return api_wrapper.run(query)
    except Exception as e:
        # Capturamos errores como: artículo no encontrado, problemas de red, etc.
        return f"Error buscando en Wikipedia: {str(e)}"


# Definimos la herramienta de DuckDuckGo.
# Esta herramienta permite al agente buscar en internet para obtener
# información actualizada que no está en su conocimiento interno.
#
# Parámetros:
# - query: str -> La consulta de búsqueda
#
# Retorna:
# - str -> Los resultados de la búsqueda formateados como texto
@tool
def duckduckgo_tool(query: str) -> str:
    """Útil cuando necesitas realizar una búsqueda en internet para encontrar información actualizada."""
    try:
        # Creamos una instancia del buscador DuckDuckGo.
        # DuckDuckGo es gratuito y no requiere API key, lo que lo hace
        # ideal para pruebas y desarrollo.
        search = DuckDuckGoSearchRun()
        # Ejecutamos la búsqueda y devolvemos los resultados.
        # DuckDuckGo devuelve los primeros resultados relevantes
        # formateados como texto legible.
        return search.run(query)
    except Exception as e:
        # Capturamos errores como: problemas de red, rate limiting, etc.
        return f"Error en búsqueda DuckDuckGo: {str(e)}"


# Creamos la lista de herramientas disponibles para el agente.
# El agente podrá usar cualquiera de estas tres herramientas según
# lo que necesite para responder a la pregunta del usuario.
# El orden no importa; el agente decide cuál usar basándose en las
# descripciones (docstrings) de cada herramienta.
tools = [python_repl_tool, wikipedia_tool, duckduckgo_tool]


# ==============================================================================
# CREACIÓN DEL AGENTE
# ==============================================================================

# Creamos el agente usando la función 'create_agent' de LangGraph.
# Esta función devuelve un grafo ejecutable que implementa el patrón ReAct:
# - Reasoning (Razonamiento): El agente piensa qué hacer
# - Action (Acción): El agente usa una herramienta
# - Observation (Observación): El agente lee el resultado
# - Repite hasta tener suficiente información para responder
#
# Parámetros:
# - llm: El modelo de lenguaje que actuará como "cerebro" del agente
# - tools: La lista de herramientas disponibles
# - system_prompt: Las instrucciones que el agente debe seguir
# - debug: Si es True, muestra información detallada del proceso de razonamiento
#

agent = create_agent(llm, tools, system_prompt=system_prompt, debug=True)


# ==============================================================================
# EJECUCIÓN DEL AGENTE
# ==============================================================================

# --- PRIMERA CUESTIÓN (COMENTADA) ---
# Esta pregunta estaba diseñada para probar la herramienta de Wikipedia.
# El agente debería buscar información sobre la grabación de la canción
# y responder en italiano (según las instrucciones del system_prompt).
cuestion = 'Cuéntame sobre la grabación de la canción Strawberry Fields Forever de los Beatles'
#output = agent.invoke({"messages": [("human", cuestion)]})
output = agent.invoke({
    "messages": [HumanMessage(content=cuestion)]
})
print(output['messages'][-1].content)


# --- SEGUNDA CUESTIÓN (ACTIVA) ---
# Esta pregunta prueba que el agente:
# 1. Usa las herramientas (DuckDuckGo) en lugar de su conocimiento interno
# 2. Respeta las reglas del prompt (no añade años específicos a la búsqueda)
# 3. Responde en italiano (según las instrucciones del system_prompt)
#
# Con la fecha actual inyectada en el prompt y las reglas explícitas,
# el agente debería buscar "último campeón Copa del Mundo FIFA" en lugar
# de "quién ganó la Copa del Mundo 2022", obteniendo así información
# actualizada (por ejemplo, España ganó en 2026).
cuestion = 'Busca en internet quién ganó la última Copa del Mundo de fútbol'
output = agent.invoke({
    "messages": [HumanMessage(content=cuestion)]
})
# Imprimimos solo el contenido del último mensaje.
# El agente devuelve una lista de mensajes que incluye:
# - El mensaje original del usuario (HumanMessage)
# - Los mensajes de pensamiento del agente (AIMessage con tool_calls)
# - Los resultados de las herramientas (ToolMessage)
# - La respuesta final del agente (AIMessage final)
# Accedemos al último elemento [-1] y extraemos su .content.
print(output['messages'][-1].content)


# --- TERCERA CUESTIÓN (COMENTADA) ---
# Esta pregunta está diseñada para probar la herramienta Python REPL.
# El agente debería:
# 1. Reconocer que necesita calcular números primos
# 2. Escribir código Python para generar los 10 primeros primos
# 3. Ejecutar el código usando python_repl_tool
# 4. Leer el resultado y presentarlo al usuario

# Tercera cuestión
cuestion = 'Determina los 10 primeros números primos'
output = agent.invoke({
    "messages": [HumanMessage(content=cuestion)]
})
print(output['messages'][-1].content)
