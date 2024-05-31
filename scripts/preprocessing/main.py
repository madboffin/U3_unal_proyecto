import preprocess_df as ppd
import preprocess_text as ppt
import pandas as pd
from pathlib import Path


def main():

    input_file = r"src\txt_comments\jigsaw_data.zip"
    output_file = r"src\txt_comments\jigsaw_data_prep.parquet"
    output_file_sampled = r"src\txt_comments\jigsaw_data_prep_sampled.parquet"
    df = pd.read_csv(input_file)
    df = ppd.preprocess_df(df, ["toxicity"])
    df.to_parquet(output_file, index=False, compression="gzip")

    sampled = ppd.sample_with_quota(df, n_by_quota=5_000)
    sampled["comment_doc"] = ppt.get_spacy_doc(sampled["comment_text"])
    sampled["comment_vec"] = sampled["comment_doc"].apply(lambda x: x.vector)
    sampled["comment_text_prep"] = sampled["comment_doc"].apply(ppt.preprocess_text)
    sampled = sampled.drop(columns=["comment_doc"])
    sampled.to_parquet(output_file_sampled, index=False, compression="gzip")


if __name__ == "__main__":
    main()
