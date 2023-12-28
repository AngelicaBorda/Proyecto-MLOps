
from fastapi import FastAPI
import pandas as pd



data_userforgenre = pd.read_csv("./data/User_For_Genres.csv")  # importo mis dataset

Most_Played_Genre = pd.read_csv("./data/Most_Played_Genre.csv")

Top_Recommended_Games = pd.read_csv("./data/Top_Recommended_Games.csv")

app = FastAPI()  # instancio la API

####### Funcion 1


@app.get("/Most_Played_Genre{genero}")
def PlayTimeGenre(genero: str = None) -> dict:

    '''Devuelve el año de lanzamiento con más horas jugadas para el género dado

    Args:
        genero (str, opcional): inserte un género. Defaults None.

    Return:
        dict: retorna un diccionario

    '''
    df_genero = Most_Played_Genre[Most_Played_Genre['generos'] == genero]

    df_suma_horas = df_genero.groupby('release_date')['playtime_forever'].sum()

    año_max_horas = df_suma_horas.idxmax()

    result = {"Año de lanzamiento con más horas jugadas para el género {}".format (genero): año_max_horas}
    return result



#######Funcion 2

@app.get("/User_For_Genres")  # la ruta es el endpoint
def get_user_for_genre(genero: str = None) -> dict:
    ''' devolver usuario que mas horas jugo un genero 

    Args:
        genero(str, opcional): inserte un genero. Defaults None.

    Return:
        dict: retorna un diccionario

    '''

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
def UsersRecommend(año: int):
    
    ''' devolver para el año dado, el top 3 juegos mas recomendados 

    Args:
        año(int): inserte un año. 

    Return:
        dict: retorna un diccionario

    '''
    
    # Filtrar el dataset por el año proporcionado
    filtered_data = Top_Recommended_Games[Top_Recommended_Games['año_posted'] == año]

    # Filtrar los juegos recomendados (recommend=1 y sentimiento=1 o 2)
    recommended_games = filtered_data[(filtered_data['recommend'] == 1) & (filtered_data['sentimiento'].isin([1, 2]))]

    # Obtener el top 3 de juegos más recomendados
    top_games = recommended_games.nlargest(3, 'recommend')

    # Crear el resultado en el formato deseado
    result = [{"Puesto 1": top_games.iloc[0]['item_name']},
              {"Puesto 2": top_games.iloc[1]['item_name']},
              {"Puesto 3": top_games.iloc[2]['item_name']}]

    return result

######## Funcion 4

@app.get("/Top_Recommended_Games")
def UsersRecommendLeast(año: int):
    
    ''' devolver para el año dado, el top 3 juegos menos recomendados 

    Args:
        año(int): inserte un año. 

    Return:
        dict: retorna un diccionario

    '''
    
    # Filtrar el dataset por el año proporcionado
    filtered_data = Top_Recommended_Games[Top_Recommended_Games['año_posted'] == año]

    # Filtrar los juegos menos recomendados (recommend=0 y sentimiento=1 o 2)
    least_recommended_games = filtered_data[(filtered_data['recommend'] == 0) & (filtered_data['sentimiento'].isin(0))]

    # Obtener el top 3 de juegos menos recomendados
    bottom_games = least_recommended_games.nsmallest(3, 'recommend')

    # Crear el resultado en el formato deseado
    result = [{"Puesto 1 de juego menos recomendado": bottom_games.iloc[0]['item_name']},
              {"Puesto 2 de juego menos recomendado": bottom_games.iloc[1]['item_name']},
              {"Puesto 3 de juego menos recomendado": bottom_games.iloc[2]['item_name']}]

    return result


####### Funcion 5

