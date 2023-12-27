from fastapi import FastAPI
import pandas as pd



data_userforgenre = pd.read_csv("./data/User_For_Genres.csv")  # importo mis dataset
Most_Played_Genre = pd.read_csv("./Most_Played_Genre.csv")


app = FastAPI()  # instancio la API

####### Funcion 1

@app.get("/Most_Played_Genre")
def PlayTimeGenre(genero: str = None) -> dict: 
    """Devuelve el año de lanzamiento con mas horas jugadas para el genero dado

    Args: 
        genero (str, opcional): inserte un género. Defaults None.

    Return:
        dict: retorna un diccionario
    
    """
    df_genero = Most_Played_Genre[Most_Played_Genre['generos'] == genero]
    
    # Agrupar por año de lanzamiento y calcular la suma de las horas jugadas
    df_suma_horas = df_genero.groupby('release_date')['playtime_forever'].sum()
    
    # Obtener el año con más horas jugadas
    año_max_horas = df_suma_horas.idxmax()
    
    return {"Año de lanzamiento con más horas jugadas para el genero" + genero: año_max_horas}



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


