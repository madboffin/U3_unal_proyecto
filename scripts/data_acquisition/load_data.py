from pathlib import Path
import pandas as pd
import gdown

def load_data(download_data: bool = False, output_file=None) -> pd.DataFrame:
    file_id = "1D31Z6sUCVUynRyimxs2n8rrULnlwhP9l"
    url = f"https://drive.google.com/uc?id={file_id}"
    if output_file is None:
        output_file = Path("src/txt_comments/jigsaw_data.zip")
        print(f"Descargando el dataset en {output_file.resolve()}")
    else:
        output_file = Path(output_file)
    if download_data:
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
        gdown.download(url, str(output_file))

    # Leer el archivo CSV desde el archivo ZIP
    df = pd.read_csv(output_file, compression='zip', dtype=str)
    print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
    
    # Reducir el dataset a 10,000 filas
    df_reducido = df.sample(n=10000, random_state=1)
    print(f"El dataset reducido tiene {df_reducido.shape[0]} filas y {df_reducido.shape[1]} columnas.")
    
    return df_reducido

# Llamada a la función para cargar los datos y reducirlos
df = load_data(download_data=True)