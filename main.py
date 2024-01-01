
from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import Dict
from fastapi.responses import JSONResponse
import numpy as np
import json
from typing import Optional, List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer



data_userforgenre = pd.read_csv("./data/User_For_Genres.csv")  # importo mis dataset

Most_Played_Genre = pd.read_csv("./data/Most_Played_Genre.csv")

Top_Recommended_Games = pd.read_csv("./data/Top_Recommended_Games.csv")

Top_Less_Recommended = pd.read_csv("./data/Top_Recommended_Games.csv")

User_Sentiment = pd.read_csv("./data/User_Sentiment.csv")

data_model_sample = pd.read_csv("./data/Data_Model_sample.csv")


app = FastAPI()  # instancio la API

####### Funcion 1

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json
import numpy as np
import pandas as pd

app = FastAPI()

# Supongamos que `Most_Played_Genre` es un DataFrame definido previamente

# Manejar la serialización de numpy.int64
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

# Uso de un Codificador Personalizado para la respuesta JSON
@app.get("/Most_Played_Genre")
def PlayTimeGenreCustom(genero: str = None) -> JSONResponse:
    # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo y ajusta la ruta según sea necesario
    archivo_csv = "data/Most_Played_Genre.csv"

    # Cargar el conjunto de datos desde el archivo CSV
    my_dataset = pd.read_csv(archivo_csv)

    # Filtrar por género
    df_genero = my_dataset[my_dataset['generos'] == genero]

    # Calcular las horas totales jugadas por fecha de lanzamiento
    df_suma_horas = df_genero.groupby('release_date')['playtime_forever'].sum()

    # Encontrar el año con más horas jugadas
    año_max_horas = df_suma_horas.idxmax()

    result = {"Año de lanzamiento con más horas jugadas para el género {}".format(genero): año_max_horas}

    # Convertir el valor de `año_max_horas` a un tipo de dato serializable (int)
    result["Año de lanzamiento con más horas jugadas para el género {}".format(genero)] = int(año_max_horas)

    # Utilizar el Codificador Personalizado con ensure_ascii=False
    content = json.dumps(result, ensure_ascii=False)
    return JSONResponse(content=content)



   
#######Funcion 2

@app.get("/User_For_Genres")
def get_user_for_genre(genero: Optional[str] = None) -> dict:
    ''' devolver usuario que mas horas jugo un genero 

    Args:
        genero (Optional[str]): inserte un genero. Defaults None.

    Return:
        dict: retorna un diccionario

    '''

    # Verificar si se proporciona un género
    if genero is None:
        return "Por favor, ingrese un género válido."

    # Verificar si el género ingresado está en el dataset
    if genero not in data_userforgenre['generos'].unique():
        return "El género ingresado es incorrecto."

    df_filtered_by_genre = data_userforgenre[data_userforgenre['generos'] == genero]
    user_hours = df_filtered_by_genre.groupby('user_id')['playtime_forever'].sum()
    max_time_played_user = user_hours.idxmax()

    max_user_df = df_filtered_by_genre[df_filtered_by_genre['user_id'] == max_time_played_user]
    hours_by_year = max_user_df.groupby('release_date')['playtime_forever'].sum().reset_index()
    hours_years_list = [
        {'Año':year, 'Horas': hours} for year, hours in zip(hours_by_year['release_date'], hours_by_year['playtime_forever'])
    ]
    result = {
        'Usuario con más horas jugadas para el género {}'.format(genero): max_time_played_user,
        'Horas jugadas': hours_years_list
    }

    return result

####### Funcion 3

@app.get("/Top_Recommended_Games")
def UsersRecommend(año: Optional[int]):
    ''' devolver para el año dado, el top 3 de juegos mas recomendados 

    Args:
        año (Optional[int]): inserte un año. 

    Return:
        dict: retorna un diccionario

    '''
    
    # Verificar si se proporciona un año
    if año is None:
        return "Por favor, ingrese un año válido."

    # Filtrar el dataset por el año proporcionado
    filtered_data = Top_Recommended_Games[Top_Recommended_Games['año_posted'] == año]

    # Verificar si hay datos para el año ingresado
    if filtered_data.empty:
        return "No hay datos para el año ingresado."

    # Filtrar los juegos recomendados (recommend=1 y sentimiento=1 o 2)
    recommended_games = filtered_data[(filtered_data['recommend'] == 1) & (filtered_data['sentimiento'].isin([1, 2]))]

    # Verificar si hay datos para los juegos recomendados
    if recommended_games.empty:
        return "No hay juegos recomendados para el año ingresado."

    # Obtener el top 3 de juegos más recomendados
    top_games = recommended_games.nlargest(3, 'recommend')

    # Crear el resultado en el formato deseado
    result = [{"Puesto 1": top_games.iloc[0]['item_name']},
              {"Puesto 2": top_games.iloc[1]['item_name']},
              {"Puesto 3": top_games.iloc[2]['item_name']}]

    return result

######## Funcion 4


@app.get("/Top_Less_Recommended")
def UsersRecommendLeast(año: Optional[int]):
    ''' devolver para el año dado, el top 3 de juegos menos recomendados 

    Args:
        año (Optional[int]): inserte un año. 

    Return:
        dict: retorna un diccionario

    '''
    
    # Verificar si se proporciona un año
    if año is None:
        return "Por favor, ingrese un año válido."

    # Filtrar el dataset por el año proporcionado
    filtered_data = Top_Recommended_Games[Top_Recommended_Games['año_posted'] == año]

    # Verificar si hay datos para el año ingresado
    if filtered_data.empty:
        return "No hay datos para el año ingresado."

    # Filtrar los juegos menos recomendados (recommend=0 y sentimiento=1 o 2)
    least_recommended_games = filtered_data[(filtered_data['recommend'] == 0) & (filtered_data['sentimiento'].isin([0]))]

    # Verificar si hay datos para los juegos menos recomendados
    if least_recommended_games.empty:
        return "No hay juegos menos recomendados para el año ingresado."

    # Obtener el top 3 de juegos menos recomendados
    bottom_games = least_recommended_games.nsmallest(3, 'recommend')

    # Crear el resultado en el formato deseado
    result = [{"Puesto 1 de juego menos recomendado": bottom_games.iloc[0]['item_name']},
              {"Puesto 2 de juego menos recomendado": bottom_games.iloc[1]['item_name']},
              {"Puesto 3 de juego menos recomendado": bottom_games.iloc[2]['item_name']}]

    return result


####### Funcion 5



@app.get("/User_Sentiment")
def sentiment_analysis(año: Optional[int] = None):
    # Filtra el dataset por el año proporcionado
    filtered_data = User_Sentiment[User_Sentiment['release_date'] == año]

    if filtered_data.empty:
        result = "No hay datos para el año ingresado."
    else:
        # Crea el diccionario de retorno
        result = {
            "Negative": len(filtered_data[filtered_data['sentimiento'] == 0]),
            "Neutral": len(filtered_data[filtered_data['sentimiento'] == 1]),
            "Positive": len(filtered_data[filtered_data['sentimiento'] == 2])
        }

    return result


####### MODELO RECOMENDACION

# Creamos la matriz de utilidad, que tiene usuarios en las filas, juegos en las columnas y el tiempo jugado como los valores.
utility_matrix = pd.pivot_table(data_model_sample, values='playtime_forever', index='user_id', columns='item_name', fill_value=0)

# Normalizamos la matriz para que las diferencias en las magnitudes de las horas de juego no afecten la similitud
utility_matrix_norm = utility_matrix.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# Manejamos los NaN rellenándolos con la media de cada columna, creamos un objeto SimpleImputer de scikit-learn y lo utilizamos para rellenar los valores Nan
#Esto asegura que no haya valores faltantes antes de aplicar PCA.
imputer = SimpleImputer(strategy='mean')
utility_matrix_norm_imputed = pd.DataFrame(imputer.fit_transform(utility_matrix_norm), columns=utility_matrix_norm.columns)

# Utilizamos PCA (Análisis de Componentes Principales) para reducir la dimensionalidad de los datos a 100 componentes principales.

pca = PCA(n_components=100)
#fit_transform ajusta el modelo y reduce las dimensiones de la matriz normalizada e imputada.
#cosine_similarity calcula la similitud coseno entre las columnas de la matriz reducida.
cosine_sim_pca = cosine_similarity(pca.fit_transform(utility_matrix_norm_imputed.T)) 

# Verificamos que las longitudes coincidan después de la reducción de dimensionalidad.
#Esto es una medida de seguridad para asegurarnos de que los datos estén bien formateados y que la operación de PCA haya tenido éxito.
if len(cosine_sim_pca) != len(utility_matrix.columns):
    raise ValueError("Las longitudes de cosine_sim_matrix y utility_matrix.columns no coinciden.")


@app.get("/Data_Model_sample")
async def recommend_games_api(input_game: str, n: Optional[int] = 5):
    '''
    Devuelve juegos recomendados basados en similitud
    
    Args:
        input_game(str): El juego de entrada.
        n(int): Número de juegos recomendados, por defecto 5.

    Return: 
        dict: Lista de juegos recomendados.
    '''
    
    # Verificar si el juego de entrada está en la base de datos
    if input_game not in utility_matrix.columns:
        raise HTTPException(status_code=404, detail=f"El juego '{input_game}' no se encuentra en la base de datos.")

    # Encuentra el índice del juego de entrada
    try:
        game_index = utility_matrix.columns.get_loc(input_game)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"El juego '{input_game}' no se encuentra en la base de datos.")

    # Calcula la similitud del juego de entrada con otros juegos
    sim_scores = list(enumerate(cosine_sim_pca[game_index]))

    # Ordena los juegos según la similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Toma los primeros n juegos recomendados (excluyendo el juego de entrada)
    top_games = sim_scores[1:n+1]

    # Obtiene los nombres de los juegos recomendados
    recommended_games = [utility_matrix.columns[i[0]] for i in top_games]

    return {"input_game": input_game, "recommended_games": recommended_games}