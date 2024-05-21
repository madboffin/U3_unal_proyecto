import preprocess_df as ppd
import preprocess_text as ppt
import pandas as pd
from pathlib import Path


def main():
    input_file = r"src\txt_comments\jigsaw_data.zip"
    output_file = r"src\txt_comments\jigsaw_data_prep.parquet"
    df = pd.read_csv(input_file)
    df = ppd.preprocess_df(df, ["toxicity"])
    df.to_parquet(output_file, index=False, compression="gzip")


if __name__ == "__main__":
    main()
