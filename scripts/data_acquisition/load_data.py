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

    df = pd.read_csv(output_file, dtype=str)
    print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
