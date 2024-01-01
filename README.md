
![](https://github.com/AngelicaBorda/Proyecto-MLOps/blob/main/mlops%20%C3%ADtulo.png)



# Proyecto MLOps 

## Recomendación de Videojuegos para Usuarios

### Descripción del Problema

##### Rol a Desarrollar

Steam, una plataforma multinacional de videojuegos, me ha asignado la tarea de crear un sistema de recomendación de videojuegos para usuarios. El desafío es trabajar desde cero y tener un Producto Mínimo Viable (MVP) al final del proyecto.

### - **Notebook: EDA_Feature-Engineering**

En el archivo Jupyter se llevó a cabo la limpieza, transformación y análisis exploratorio de datos. 
También se aplicó análisis de sentimiento con NLP, a la columna reviews, que se encontraba en el dataset User_Review y se la reemplazó por la columna Sentimiento, la cual tiene los valores 0 si la reseña era mala, 1 neutra y 2 positiva. Y en caso de reseña ausente se deja el valor 1.


### - **Carpeta Data**

<p>Se crearon datasets preparados para cada función y para el modelo de recomendación, esto con motivo de agilizar el procesamiento de datos y evitar errores al momento del deploy en render.</p>

 <p>1. User_For_Genres.csv</p>

<p>2. Most_Played_Genre.csv</p>

<p>3. Top_Recommended_Games.csv</p>

<p>4. User_Sentiment.csv</p>

<p>5. Data_Model_sample.csv</p>

## - **Archivo Main**

 <p>Contiene las funciones para los endpoints que se consumirán en la API.</pp>



### <p>@app.get("/most_played_genre")
def PlayTimeGenreCustom(genero: str = None) -> JSONResponse:</p>


<p>Para el género ingresado devuelve el año con más horas jugadas por los usuarios.</p>



### <p>@app.get("/User_For_Genres")
def get_user_for_genre(genero: Optional[str] = None) -> dict:</p>


<p>Para el género ingresado devuelve el usuario que acumula mas horas jugadas y una lista de horas acumuladas por año.</p>



### <p>@app.get("/Top_Recommended_Games")
def UsersRecommend(año: Optional[int]):</p></p>


<p>Para el año ingresado, devuelve el top 3 de juegos más recomendados.</p>



### <p>@app.get("/Top_Less_Recommended")
def UsersRecommendLeast(año: Optional[int]):</p>


<p>Para el año ingresado, devuelve el top 3 de juegos menos recomendados.</p>



### <p>@app.get("/User_Sentiment")
def sentiment_analysis(año: Optional[int] = None):</p></p>


<p>Para el año ingresado, devuelve una lista con la cantidad de reseñas de los usuarios. Categorizadas con análisis de sentimiento. </p></p>



## - Sistema de Recomendación 

<p>También se encuentra en el archivo main.</p>


<p>El  Modelo e basa en una relación Item-Item y sigue los siguientes pasos:</p>

### **Creación de la Matriz de Utilidad:**
<p>Se genera una matriz que tiene usuarios en las filas, juegos en las columnas y el tiempo jugado como valores.</p>

### **Normalización de la Matriz:**
<p>La matriz se normaliza para mitigar las diferencias en las magnitudes de las horas de juego.</p>

### **Manejo de NaN e Imputación con PCA:**
<p>Los NaN se manejan rellenándolos con la media de cada columna.
Se utiliza PCA (Análisis de Componentes Principales) para reducir la dimensionalidad de los datos a 100 componentes principales.</p>

### **Verificación de Longitudes después de la Reducción de Dimensionalidad:**
<p>Se verifica que las longitudes coincidan después de la reducción de dimensionalidad para garantizar el formato correcto de los datos.</p>

### **API para Recomendación de Juegos:**
<p>Se expone una API con un endpoint para recomendar juegos basados en la similitud del juego de entrada.</p>


<p>Este modelo ofrece recomendaciones de juegos basadas en la similitud de los patrones de juego, proporcionando una experiencia personalizada a los usuarios.</p>
