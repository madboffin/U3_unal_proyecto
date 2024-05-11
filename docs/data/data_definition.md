# Definición de los datos

## Origen de los datos

- El dataset fue obtenido de la plataforma "Kaggle" y es una recopilación de mensajes de twitter que están asociados a variables como un id, fecha de publicación, puntaje y variables de clasificación de toxicidad. Este dataset se encuentra como parte de una competición donde se busca crear un modelo de machine learning que pueda predecir cuales Tweets son verdaderos "desastres" y cuales no lo son. La creación del dataset no es muy clara ya que en kaggle se referencia una pagina web que ya no está en funcionamiento.

## Especificación de los scripts para la carga de datos

Para la elaboración del proyecto se utilizaron los siguientes scripts:

- os: se utiliza para interactuar con el sistema operativo. En este caso, se utiliza para crear directorios.

- gdown: es una libreria que permite descargar archivos de Google Drive mediante su URL. Se utiliza para descargar el archivo de datos del enlace proporcionado.

- pandas: se utiliza para cargar, manipular los datos tabulares. En este caso se utiliza 'pd.read_csv' para cargar el archivo CSV descargado en un DataFrame de Pandas.

## Referencias a rutas o bases de datos origen y destino

- La ruta de la base de datos de origen utilizada es https://drive.google.com/uc?id=1D31Z6sUCVUynRyimxs2n8rrULnlwhP9l donde se encuentra un archivo .zip con el csv donde se almacena la base de datos que se va a utilizar en el proyecto.

- La ruta de base de destino se encuentra en el repositorio de Github en la carpeta database en el link https://github.com/madboffin/U3_unal_proyecto/tree/master/src/nombre_paquete/database.

### Rutas de origen de datos

- El archivo de origen de los datos está ubicado en un link público de Google Drive que será transformado para utilizar únicamente los datos relevantes para el proyecto.

- Los datos originalmente están en formato csv como un data frame.

- La transformación de los datos que se va a realizar para el proyecto va a ser eliminar todas las columnas que sean diferentes a los mensajes de twitter y su clasificación de toxicidad. También se van a eliminar todas las filas que tengan datos vacios en alguna de las columnas.

### Base de datos de destino

- [ ] Especificar la base de datos de destino para los datos.
- [ ] Especificar la estructura de la base de datos de destino.
- [ ] Describir los procedimientos de carga y transformación de los datos en la base de datos de destino.
