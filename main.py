from fastapi import FastAPI
import pandas as pd

##############FUNCION 2

data_userforgenre = pd.read_csv("./data/data_funcion2.csv")  #importo mi dataset

app = FastAPI()  #instancio la API

@app.get("data_funcion2")  #la ruta es el endpoint
def get_user_for_genre(genero: str = None) -> dict:
    ''' devolver usuario que mas horas jugo un genero dado

    Args:
        genero(str, opcional): inserte un genero. Defaults None.

    Return:
        dict: retorna un diccionario

    '''

    df_filtered_by_genre = data_userforgenre[data_userforgenre['generos'] == genero]
    user_hours = df_filtered_by_genre.groupby('user_id')['playtime_forever'].sum()
    max_time_played_user = user_hours.idxmax()

    return {'Usuario con más horas jugadas para el género{}'.format(genero): max_time_played_user}
