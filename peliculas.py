# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y MÓDULOS
# ==============================================================================

# 'os' nos permite interactuar con el sistema operativo, especialmente para leer
# variables de entorno (donde guardamos claves API de forma segura).
import os

# 'requests' es una librería para hacer peticiones HTTP (GET, POST, etc.) a APIs externas.
# La usaremos para consultar la API de The Movie Database (TMDB).
import requests

# 'load_dotenv' carga las variables definidas en un archivo .env hacia el entorno de Python.
# Esto evita tener que escribir claves API directamente en el código.
from dotenv import load_dotenv

# 'ChatPromptTemplate' es la clase de LangChain que nos permite crear prompts estructurados
# con roles (system, human, ai) para modelos de chat modernos.
from langchain_core.prompts import ChatPromptTemplate

# 'ChatGroq' es el wrapper específico de LangChain para conectarse a la API de Groq.
# Groq es una plataforma que ofrece inferencia ultrarrápida de modelos de lenguaje.
from langchain_groq import ChatGroq

# ==============================================================================
# 2. DEFINICIÓN DE LA FUNCIÓN PARA CONSULTAR LA API DE TMDB
# ==============================================================================

def peliculas(pelicula):
    """
    Función que consulta la API de The Movie Database (TMDB) para buscar películas.

    Parámetros:
    - pelicula (str): El nombre de la película a buscar (puede incluir %20 para espacios)

    Retorna:
    - dict: Un diccionario con la respuesta JSON de la API, que contiene información
            sobre las películas encontradas (título, fecha, resumen, etc.)
    """

    # Construimos la URL de la API.
    # 'query={pelicula}' es el parámetro de búsqueda.
    # 'include_adult=false' excluye contenido para adultos.
    # 'language=en-US' establece el idioma de la respuesta (inglés de EE.UU.).
    # 'page=1' indica que queremos la primera página de resultados.
    url = f"https://api.themoviedb.org/3/search/movie?query={pelicula}&include_adult=false&language=en-US&page=1"

    # Los headers (cabeceras) de la petición HTTP.
    # 'accept: application/json' le dice a la API que queremos recibir datos en formato JSON.
    # 'Authorization: Bearer ...' es el token de autenticación que nos da acceso a la API.
    headers = {
        "accept": "application/json",
        # ⚠️ SEGURIDAD: El token debería estar en el archivo .env, no hardcodeado aquí.
        # Lo movemos a una variable de entorno para mayor seguridad.
        "Authorization": f"Bearer {tmdb_token}"
    }

    # Realizamos la petición GET a la API de TMDB.
    # 'requests.get()' envía la petición HTTP con la URL y los headers especificados.
    response = requests.get(url, headers=headers)

    # Convertimos la respuesta JSON a un diccionario de Python y la retornamos.
    # Este diccionario contendrá una lista de películas con toda su información.
    return response.json()


# ==============================================================================
# 3. CARGA DE VARIABLES DE ENTORNO Y CONFIGURACIÓN
# ==============================================================================

# Carga el archivo .env y hace que sus variables estén disponibles en os.environ.
# Esto es crucial para no exponer claves API en el código fuente.
load_dotenv()

# Obtenemos la clave API de Groq desde las variables de entorno.
# NOTA: ChatGroq la detectará automáticamente si se llama exactamente "GROQ_API_KEY",
# pero la guardamos en una variable por si necesitamos usarla en otro lugar del código.
api_key = os.environ.get("GROQ_API_KEY")

# Obtenemos el token de TMDB desde las variables de entorno.
# ⚠️ IMPORTANTE: Debes añadir esta línea a tu archivo .env:
# TMDB_API_TOKEN=eyJhbGciOiJIUzI1NiJ9... (tu token completo)
tmdb_token = os.environ.get("TMDB_API_TOKEN")


# ==============================================================================
# 4. INICIALIZACIÓN DEL MODELO DE LENGUAJE (LLM)
# ==============================================================================

# Creamos la instancia del modelo usando ChatGroq.
# ChatGroq actúa como puente entre LangChain y la infraestructura de Groq.
llm = ChatGroq(
    # Especificamos qué modelo queremos usar.
    model="openai/gpt-oss-safeguard-20b",

    # 'temperature' controla la aleatoriedad/creatividad de las respuestas.
    # 0 = Respuestas deterministas, lógicas y predecibles (ideal para tareas de extracción/resumen).
    # 0.7 = Balance entre creatividad y coherencia.
    # 1.0 = Máxima creatividad (puede ser menos preciso).
    temperature=0,
)


# ==============================================================================
# 5. CREACIÓN DEL PROMPT ESTRUCTURADO
# ==============================================================================

# Definimos el mensaje de sistema directamente como una cadena.
# Este mensaje establece las reglas y el contexto para el asistente.
system_template = '''
Te voy a dar información sobre algunas películas en formato JSON.
Me tienes que devolver la información en español del título, fecha de estreno y resumen
de las primeras 3 que aparezcan de forma estructurada.
Si aparecen menos, me das las que aparezcan.
'''

# Creamos el prompt usando ChatPromptTemplate.
# A diferencia de PromptTemplate (texto plano), ChatPromptTemplate estructura la conversación
# en mensajes con roles, lo cual es el estándar para modelos de chat modernos.
prompt = ChatPromptTemplate.from_messages([

    # Primer mensaje: Rol "system" (Sistema).
    # Define la personalidad del asistente y las reglas generales.
    # Aquí le decimos qué debe hacer con los datos que le pasaremos.
    ("system", system_template),

    # Segundo mensaje: Rol "human" (Humano/Usuario).
    # Representa la entrada del usuario. Aquí inyectamos el JSON de películas.
    # La variable {pelicula} será reemplazada por el diccionario real cuando invoquemos la cadena.
    ("human", "{pelicula}"),
])


# ==============================================================================
# 6. CREACIÓN DE LA CADENA USANDO LCEL (LangChain Expression Language)
# ==============================================================================

# El operador "|" (pipe/tubería) conecta componentes en LCEL.
# Significa: "Toma la salida del componente izquierdo y pásala como entrada al derecho".
#
# Flujo de datos:
# 1. chain.invoke() recibe un diccionario {"pelicula": {...}}
# 2. El diccionario pasa al 'prompt', que rellena {pelicula} con el JSON
# 3. El prompt genera una lista de mensajes: [MensajeSystem, MensajeHuman]
# 4. Esa lista viaja a través del "|" hacia el 'llm'
# 5. El 'llm' procesa los mensajes y genera una respuesta
# 6. La respuesta se devuelve como un objeto AIMessage
#
# Esto reemplaza a la antigua clase LLMChain, que ahora está obsoleta.
chain = prompt | llm


# ==============================================================================
# 7. EJECUCIÓN DE LA CADENA Y OBTENCIÓN DEL RESULTADO
# ==============================================================================

# Primero, llamamos a nuestra función para obtener los datos de la película desde TMDB.
# "el%20septimo%20sello" es "El séptimo sello" con los espacios codificados como %20.
# La función devuelve un diccionario con la información de las películas encontradas.
datos_pelicula = peliculas("el%20septimo%20sello")

# Ejecutamos la cadena con .invoke().
# Le pasamos un diccionario con los valores para las variables definidas en el prompt.
# En este caso, solo tenemos {pelicula}, que será reemplazado por el JSON completo.
result = chain.invoke({
    "pelicula": datos_pelicula,  # El JSON de TMDB se inyecta aquí
})

# La variable 'result' NO es un string plano.
# Es un objeto de tipo 'AIMessage' (Mensaje de Inteligencia Artificial) que contiene:
# - .content: El texto generado por el modelo (lo que queremos mostrar)
# - .response_metadata: Metadatos como tokens usados, tiempo de respuesta, etc.
# - .additional_kwargs: Información adicional del modelo
#
# Para imprimir solo el texto de la respuesta (sin metadatos), accedemos a .content.
print(result.content)
