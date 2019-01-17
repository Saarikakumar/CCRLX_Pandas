import os
import pandas as pd
import pylab as plt

SQ_NAME = "parkin_sq.tab"
MQ_NAME = "parkin_mq.tab"
VD_NAME = "vdlist"

WORK_DIR = os.getcwd()

SQ_PATH = WORK_DIR+"/"+SQ_NAME
MQ_PATH = WORK_DIR+"/"+MQ_NAME
VD_NAME = WORK_DIR+"/"+VD_NAME


def open_data(data_path):
    with open(data_path, "r") as f:
        contents = f.readlines()
    return contents


def get_data(data_path):
    contents = open_data(data_path)
    column_names = get_column_names(contents)
    return column_names


def clean_vdlist(vdlist):
    clean_list = [float(entry.strip("s\n")) for entry in vdlist]
    return clean_list


def get_duplicates(vdlist):
    # Get unique values
    delays = list(set(VDLIST))
    delays.sort()
    duplicates = vdlist[len(delays):]
    return delays, duplicates


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


def get_heights(data_frame, column_names):
    for name in column_names:
        for index, row in data_frame.iterrows():
            data_frame.at[index, name] = data_frame.loc[index, name] * \
                                         data_frame.loc[index, "HEIGHT"]
    return data_frame


def scale_by_scans(data_frame, column_names, sq_ns, mq_ns):
    ratio = float(sq_ns/mq_ns)
    for name in column_names:
        for index, row in data_frame.iterrows():
            data_frame.at[index, name] = data_frame.loc[index, name] * ratio
    return data_frame


def divide_all(sq_df, mq_df, column_names, sq_nc, mq_nc):
    sq_df = sq_df.sort_values(by=["ASS"])
    mq_df = mq_df.sort_values(by=["ASS"])
    sq_df = sq_df.reset_index(drop=True)
    mq_df = mq_df.reset_index(drop=True)
    ratio_df = sq_df
    nc_proc = (2.0 ** (-1.0 * mq_nc)) / (2.0 ** (-1.0 * sq_nc))
    for i, name in enumerate(column_names):
        for index, row in mq_df.iterrows():
            if sq_df.loc[index, "ASS"] == mq_df.loc[index, "ASS"]:
                ratio_df.at[index, name] = mq_df.loc[index, name] \
                                        / sq_df.loc[index, name] / nc_proc
            else:
                print("ERROR: DATA MISMATCH")
    return ratio_df

RATIO = 32.0/120.0

VDLIST = open_data(VD_NAME)
VDLIST = clean_vdlist(VDLIST)

DELAYS, DUPLICATES = get_duplicates(VDLIST)

sq_df, sq_names = text_to_df(SQ_PATH)
mq_df, mq_names = text_to_df(MQ_PATH)


def reverse_mq(mq_df, delays, duplicates, names):
    column_index = list(range(2, 2+len(delays)))
    column_index.reverse()
    new_mq_df = mq_df.iloc[:, 0:2]
    mq_data = mq_df.iloc[:, column_index]
    sum_all = len(delays)+len(duplicates)
    mq_dup = mq_df.iloc[:, sum_all:2+sum_all]
    new_mq_df = new_mq_df.join(mq_data)
    new_mq_df = new_mq_df.join(mq_dup)
    new_mq_df.columns = names
    return new_mq_df

mq_df = reverse_mq(mq_df, DELAYS, DUPLICATES, list(sq_df))

print(mq_df)

"""
new_mq_columns = list(range(2,10,1))
new_mq_columns.reverse()

new_mq_df = mq_df.iloc[:,0:2]
mq_data = mq_df.iloc[:,new_mq_columns]
mq_dup = mq_df.iloc[:,10:12]
new_mq_df = new_mq_df.join(mq_data)
new_mq_df = new_mq_df.join(mq_dup)

print(new_mq_df)
"""


sq_df = remove_none(sq_df)
sq_df = remove_duplicates(sq_df)

mq_df = remove_none(mq_df)
mq_df = remove_duplicates(mq_df)

sq_df, mq_df = remove_incomplete(sq_df, mq_df)

sq_df = get_heights(sq_df, sq_names)
mq_df = get_heights(mq_df, mq_names)

mq_df = scale_by_scans(mq_df, mq_names, 32.0, 120.0)

#test = sq_df[["ASS"]]
#test = test.reset_index(drop=True)
#for name in sq_int_names:
#    buffer = [row[name]*row["HEIGHT"] for index, row in sq_df.iterrows()]
#    buffer = pd.DataFrame(np.array(buffer).reshape(len(buffer), 1), columns=[name])
#    test = test.join(buffer)
#print(test.head())

#for name in sq_int_names:
#    for index, row in sq_df.iterrows():
#        sq_df.at[index, name] = sq_df.loc[index, name] * sq_df.loc[index, "HEIGHT"]

final = divide_all(sq_df, mq_df, sq_names, -4.0, -5.0)

test =[]
for index, row in final.iterrows():
    test = row.tolist()

    plt.plot(VDLIST, test[2:], "ok")
    plt.show()
