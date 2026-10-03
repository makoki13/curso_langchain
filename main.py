# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN DEL ENTORNO
# ==============================================================================

# Importamos 'os' para interactuar con las variables de entorno del sistema.
import os

# 'load_dotenv' carga las variables desde el archivo .env (donde guardamos la clave API).
from dotenv import load_dotenv

# Importamos el generador de prompts estructurados por roles para modelos de chat.
from langchain_core.prompts import ChatPromptTemplate

# Importamos el wrapper específico de LangChain para conectarse a Groq.
from langchain_groq import ChatGroq

# Ejecutamos load_dotenv() para que la variable GROQ_API_KEY esté disponible en el entorno.
load_dotenv()

# Obtenemos la clave API.
# NOTA: ChatGroq la detectará automáticamente, pero la guardamos por si la necesitamos luego.
api_key = os.environ.get("GROQ_API_KEY")


# ==============================================================================
# 2. INICIALIZACIÓN DEL MODELO DE LENGUAJE (LLM)
# ==============================================================================

# Creamos la instancia del modelo usando la clase ChatGroq.
llm = ChatGroq(
    # Tu modelo específico. Al funcionar, confirma que tu endpoint soporta este formato.
    model="qwen/qwen3.8-27b",

    # Temperature 0: Hace que las respuestas sean deterministas, directas y sin "creatividad".
    # Ideal para tareas de traducción donde se busca precisión exacta.
    temperature=0,
)


# ==============================================================================
# 3. CREACIÓN DEL PROMPT (LA INSTRUCCIÓN PARA EL MODELO)
# ==============================================================================

# Usamos 'ChatPromptTemplate' para estructurar la conversación por roles.
# Esto es mucho más efectivo que un prompt de texto plano para modelos de chat modernos.
prompt = ChatPromptTemplate.from_messages([

    # Rol "system": Define la personalidad y las reglas globales del asistente.
    # Las variables {idioma_entrada}, {idioma_salida} y {texto} son marcadores de posición.
    (
        "system",
        "Eres un asistente útil que traduce del {idioma_entrada} al {idioma_salida} el texto {texto}.",
    ),

    # Rol "human": Representa la entrada directa del usuario.
    # Aquí inyectamos el texto específico que queremos procesar.
    ("human", "{texto}"),
])


# ==============================================================================
# 4. CREACIÓN DE LA CADENA (LCEL - LangChain Expression Language)
# ==============================================================================

# El operador "|" (pipe) conecta los componentes.
# Flujo: El diccionario de entrada -> pasa al 'prompt' -> se convierte en mensajes -> pasa al 'llm'.
# Esto reemplaza a la antigua y obsoleta clase 'LLMChain'.
chain = prompt | llm


# ==============================================================================
# 5. EJECUCIÓN DE LA CADENA Y OBTENCIÓN DEL RESULTADO
# ==============================================================================

# '.invoke()' ejecuta la cadena de forma síncrona.
# Le pasamos un diccionario con los valores reales para las variables definidas en el prompt.
result = chain.invoke({
    "idioma_entrada": "español",
    "idioma_salida": "ruso",
    "texto": "¿Por qué es tan rápido Groq?"
})

# 'result' es un objeto de tipo 'AIMessage' (que incluye metadatos como tokens usados).
# Para imprimir solo el texto generado por el modelo, accedemos a su atributo '.content'.
print(result.content)
