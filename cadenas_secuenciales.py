# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y MÓDULOS
# ==============================================================================

# 'os' nos permite interactuar con el sistema operativo, en particular para leer
# variables de entorno donde almacenamos claves API de forma segura.
import os

# 'load_dotenv' busca un archivo llamado .env en el directorio actual y carga
# todas las variables definidas en él dentro del entorno de Python (os.environ).
# Esto evita tener que escribir claves secretas directamente en el código fuente.
from dotenv import load_dotenv

# 'ChatPromptTemplate' es la clase de LangChain que permite crear prompts
# estructurados con roles (system, human, ai) para modelos de chat modernos.
# A diferencia de 'PromptTemplate' (texto plano), este organiza la conversación
# en mensajes con roles, que es el formato que esperan los LLMs actuales.
from langchain_core.prompts import ChatPromptTemplate

# 'ChatGroq' es el wrapper específico de LangChain para conectarse a la API de Groq.
# Groq es una plataforma que ofrece inferencia ultrarrápida de modelos de lenguaje
# gracias a su hardware especializado (LPU - Language Processing Unit).
from langchain_groq import ChatGroq

# ==============================================================================
# 2. CARGA DE VARIABLES DE ENTORNO Y CONFIGURACIÓN DE CREDENCIALES
# ==============================================================================

# Ejecutamos load_dotenv() para que las variables del archivo .env estén
# disponibles en os.environ. Sin esta línea, os.environ.get() devolvería None.
# El archivo .env debe contener al menos:
#   GROQ_API_KEY=tu_clave_de_groq
#   TMDB_API_TOKEN=tu_token_de_tmdb
load_dotenv()

# Obtenemos la clave API de Groq desde las variables de entorno.
# NOTA: La clase ChatGroq busca automáticamente una variable llamada "GROQ_API_KEY"
# en el entorno, por lo que técnicamente esta línea no es estrictamente necesaria
# para que el modelo funcione. Sin embargo, la guardamos por si necesitamos
# usarla en otro lugar del código o para verificar que existe.
api_key = os.environ.get("GROQ_API_KEY")

# Obtenemos el token de la API de The Movie Database (TMDB) desde las variables
# de entorno. En este script concreto no se usa (se usaba en un ejemplo anterior
# para buscar películas), pero se mantiene cargado por si se necesita más adelante.
tmdb_token = os.environ.get("TMDB_API_TOKEN")


# ==============================================================================
# 3. INICIALIZACIÓN DEL MODELO DE LENGUAJE (LLM)
# ==============================================================================

# Creamos una instancia del modelo de lenguaje usando la clase ChatGroq.
# ChatGroq actúa como puente entre LangChain y la infraestructura de Groq,
# gestionando automáticamente la autenticación, el formato de los mensajes
# y la comunicación con la API.
llm = ChatGroq(
    # El modelo específico que queremos usar para las inferencias.
    # "openai/gpt-oss-safeguard-20b" es un modelo de 20 mil millones de parámetros.
    # El formato "proveedor/nombre-modelo" indica que Groq aloja este modelo
    # y lo sirve a través de su API compatible con OpenAI.
    model="openai/gpt-oss-safeguard-20b",

    # 'temperature' controla el nivel de aleatoriedad y creatividad en las respuestas.
    # - 0.0 = Totalmente determinista. Dada la misma entrada, siempre produce
    #         exactamente la misma salida. Ideal para tareas que requieren precisión.
    # - 0.5 = Un punto intermedio. Las respuestas son coherentes y enfocadas,
    #         pero con cierta variedad en la redacción y el estilo.
    # - 1.0 = Máxima creatividad. Las respuestas pueden ser más diversas e
    #         inesperadas, pero también menos precisas o consistentes.
    # En este caso, 0.5 es un buen equilibrio: el código generado será correcto
    # y la descripción será detallada pero no repetitiva.
    temperature=0.5,
)


# ==============================================================================
# 4. DEFINICIÓN DE LA VARIABLE DE ENTRADA
# ==============================================================================

# Definimos el concepto sobre el cual trabajará toda la cadena.
# Este valor se inyectará en el primer prompt a través de la variable {concepto}.
# El modelo usará este concepto para generar una función de Python que lo implemente.
concepto = "inercia"


# ==============================================================================
# 5. DEFINICIÓN DE LOS PROMPTS (INSTRUCCIONES PARA EL MODELO)
# ==============================================================================

# --- PRIMER PROMPT ---
# Este prompt le pide al modelo que actúe como científico y programador,
# y que genere código Python que implemente el concepto especificado.
# La instrucción "Devuelve solo el código Python, sin explicaciones adicionales"
# es crucial: garantiza que la salida sea código limpio que podamos pasar
# directamente al segundo prompt sin tener que limpiar texto extra.
prompt_template_1 = '''
Eres un científico experimentado y programador en Python.
Escribe una función que implemente el concepto de {concepto}.
Devuelve solo el código Python, sin explicaciones adicionales.
'''

# --- SEGUNDO PROMPT ---
# Este prompt le pide al modelo que analice y describa detalladamente
# una función de Python que se le proporcionará.
# La variable {codigo} será reemplazada por la salida del primer LLM
# (el código Python generado sobre la entropía).
prompt_template_2 = '''
Dada la siguiente función de Python, descríbela lo más detalladamente posible:

{codigo}
'''


# ==============================================================================
# 6. CREACIÓN DE LOS PROMPTS ESTRUCTURADOS CON ROLES
# ==============================================================================

# Creamos el primer prompt estructurado usando ChatPromptTemplate.
# 'from_messages()' recibe una lista de tuplas, donde cada tupla contiene:
#   (rol, contenido_del_mensaje)
#
# Los roles disponibles son:
#   - "system": Define la personalidad, reglas y contexto del asistente.
#               El modelo usa esto como instrucciones generales.
#   - "human": Representa la entrada del usuario. Es la pregunta o tarea directa.
#   - "ai": Representa una respuesta previa del asistente (útil en conversaciones
#           de varios turnos, aunque aquí no lo usamos).
prompt1 = ChatPromptTemplate.from_messages([
    # Mensaje de sistema: Le damos las instrucciones generales al modelo.
    # La variable {concepto} será reemplazada por "entropía" cuando invoquemos la cadena.
    ("system", prompt_template_1),

    # Mensaje del usuario: Le pasamos el concepto directamente como entrada.
    # Aunque el concepto ya está en el mensaje de sistema a través de {concepto},
    # ponerlo también aquí refuerza la instrucción y sigue el patrón estándar
    # de chat donde el usuario hace una petición explícita.
    ("human", "{concepto}"),
])

# Creamos el segundo prompt estructurado.
# Este prompt está diseñado para recibir el código generado por el primer LLM
# y producir una descripción detallada del mismo.
prompt2 = ChatPromptTemplate.from_messages([
    # Mensaje de sistema: Contiene las instrucciones y el marcador {codigo}
    # que será reemplazado por el código Python generado en el paso anterior.
    ("system", prompt_template_2),

    # Mensaje del usuario: Una instrucción directa pidiendo la descripción.
    # Este mensaje es fijo (no tiene variables) porque toda la información
    # variable ya está en el mensaje de sistema a través de {codigo}.
    ("human", "Describe el código anterior"),
])


# ==============================================================================
# 7. CREACIÓN DE LA CADENA SECUENCIAL USANDO LCEL
# ==============================================================================

# Aquí construimos la cadena completa usando LCEL (LangChain Expression Language).
# El operador "|" (pipe/tubería) conecta componentes de izquierda a derecha.
# La salida de cada componente se convierte en la entrada del siguiente.
#
# Este patrón reemplaza a la antigua clase 'SimpleSequentialChain' que estaba
# en langchain.chains y que ahora está obsoleta.
#
# FLUJO COMPLETO DE DATOS:
#
# Paso 1: prompt1
#   - Entrada:  {"concepto": "entropía"}
#   - Salida:   Lista de mensajes [SystemMessage, HumanMessage]
#               con {concepto} reemplazado por "entropía"
#
# Paso 2: llm (primera llamada)
#   - Entrada:  La lista de mensajes del paso 1
#   - Salida:   Un objeto AIMessage que contiene el código Python generado
#               Ejemplo: AIMessage(content="def calcular_entropia(p):\n    ...")
#
# Paso 3: (lambda x: x.content)
#   - Entrada:  El objeto AIMessage del paso 2
#   - Salida:   Un string con solo el texto del código Python
#               Ejemplo: "def calcular_entropia(p):\n    ..."
#   - NOTA: Este paso es NECESARIO porque el siguiente paso necesita un string,
#           no un objeto AIMessage. El atributo .content extrae el texto plano.
#
# Paso 4: (lambda x: {"codigo": x})
#   - Entrada:  El string con el código Python del paso 3
#   - Salida:   Un diccionario {"codigo": "def calcular_entropia(p):\n    ..."}
#   - NOTA: Este paso es NECESARIO porque prompt2 espera un diccionario con
#           la clave "codigo" para rellenar la variable {codigo} en el template.
#           Si pasáramos el string directamente, LangChain daría un error:
#           "The input to RunnablePassthrough.assign() must be a dict" o similar.
#
# Paso 5: prompt2
#   - Entrada:  El diccionario {"codigo": "..."} del paso 4
#   - Salida:   Lista de mensajes [SystemMessage, HumanMessage]
#               con {codigo} reemplazado por el código Python real
#
# Paso 6: llm (segunda llamada)
#   - Entrada:  La lista de mensajes del paso 5
#   - Salida:   Un objeto AIMessage con la descripción detallada del código
#
chain = (
    prompt1                             # Paso 1: Rellenar prompt con {concepto}
    | llm                               # Paso 2: Generar código Python
    | (lambda x: x.content)             # Paso 3: Extraer texto del AIMessage → string
    | (lambda x: {"codigo": x})         # Paso 4: Convertir string → dict {"codigo": "..."}
    | prompt2                           # Paso 5: Rellenar prompt con {codigo}
    | llm                               # Paso 6: Generar descripción del código
)


# ==============================================================================
# 8. EJECUCIÓN DE LA CADENA Y OBTENCIÓN DEL RESULTADO
# ==============================================================================

# Ejecutamos la cadena completa con el método .invoke().
# .invoke() es el método síncrono estándar de LCEL para ejecutar una cadena.
# Recibe un diccionario con los valores iniciales que necesita el primer prompt.
# En este caso, solo necesitamos pasar "concepto", ya que "codigo" se genera
# automáticamente en el paso intermedio de la cadena.
#
# Existen otros métodos de ejecución:
#   - .stream(): Devuelve la respuesta en chunks (fragmentos) en tiempo real.
#                Útil para mostrar la respuesta progresivamente al usuario.
#   - .ainvoke(): Versión asíncrona de .invoke() para usar con async/await.
#   - .batch(): Ejecuta la cadena con múltiples entradas a la vez.
resultado = chain.invoke({"concepto": concepto})

# La variable 'resultado' es un objeto de tipo 'AIMessage'.
# Este objeto contiene varios atributos:
#   - .content: El texto generado por el modelo (lo que queremos mostrar).
#   - .response_metadata: Metadatos de la respuesta (tokens usados, modelo, etc.).
#   - .additional_kwargs: Información adicional específica del proveedor.
#   - .type: El tipo de mensaje (en este caso, "ai").
#
# Para imprimir solo el texto limpio de la descripción generada por el modelo,
# accedemos al atributo .content. Si hiciéramos print(resultado) sin .content,
# veríamos toda la estructura del objeto AIMessage con sus metadatos.
print(resultado.content)
