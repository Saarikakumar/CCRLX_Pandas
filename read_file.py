import os
import pandas as pd
import numpy as np

SQ_NAME = "parkin_sq.tab"
MQ_NAME = "parkin_mq.tab"

WORK_DIR = os.getcwd()

SQ_PATH = WORK_DIR+"/"+SQ_NAME
MQ_PATH = WORK_DIR+"/"+MQ_NAME


def get_data(data_path):
    with open(data_path, "r") as f:
        contents = f.readlines()
        column_names = get_column_names(contents)
    return column_names


def get_column_names(data):
    for index, row in enumerate(data):
        if row.startswith("VARS"):
            column_names = row.split()[1:]
            break
    return column_names


def get_data_frame(data_path, column_names):
    data_frame = pd.read_table(data_path, names=column_names, index_col=None,
                               header=None, sep="\s+", skiprows=12)
    first_labels = data_frame[["ASS", "HEIGHT"]]
    intensity = data_frame.loc[:, data_frame.columns.str.startswith("Z_")]
    intensity_names = list(intensity)
    clean_data_frame = first_labels.join(intensity)
    data_frame = None
    return clean_data_frame, intensity_names


def text_to_df(data_path):
    column_names = get_data(data_path)
    data_frame, intensity_names = get_data_frame(data_path, column_names)
    return data_frame, intensity_names


def remove_none(data_frame):
    data_frame = data_frame[data_frame["ASS"] != "None"]
    return data_frame


def remove_duplicates(data_frame):
    data_frame = data_frame.drop_duplicates(subset="ASS", keep=False)
    return data_frame


def remove_incomplete(sq_df, mq_df):
    sq_df_clean = sq_df[sq_df["ASS"].isin(mq_df["ASS"])]
    mq_df_clean = mq_df[mq_df["ASS"].isin(sq_df["ASS"])]
    return sq_df_clean, mq_df_clean


sq_df, sq_int_names = text_to_df(SQ_PATH)
mq_df, mq_int_names = text_to_df(MQ_PATH)


sq_df = remove_none(sq_df)
sq_df = remove_duplicates(sq_df)

mq_df = remove_none(mq_df)
mq_df = remove_duplicates(mq_df)

sq_df, mq_df = remove_incomplete(sq_df, mq_df)

test = sq_df[["ASS"]]
test = test.reset_index(drop=True)
for name in sq_int_names:
    buffer = [row[name]*row["HEIGHT"] for index, row in sq_df.iterrows()]
    buffer = pd.DataFrame(np.array(buffer).reshape(len(buffer), 1), columns=[name])
    test = test.join(buffer)

print(test.shape)