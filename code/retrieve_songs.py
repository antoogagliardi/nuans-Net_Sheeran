import os
import re
import pandas as pd
from pprint import pprint
from tqdm import tqdm

from lyricsgenius import Genius 

from src.utils import extract_song, read_config_file

if __name__ == "__main__":
    # Reading configuration file
    cfg = read_config_file(config_path="./configs/config.yaml")
    pprint(cfg)

    # Paths setup
    root_folder = "../"
    print("Project Root Folder: ", root_folder)
    cwd = os.getcwd()
    print("Current work directory: ", cwd)
    data_path = os.path.join(root_folder, cfg["paths"]["data"])
    print("data path: ", data_path)


    client_acces_token = ">>| Put here your Genius API Token |<<"
    genius = Genius(access_token=client_acces_token,
                    response_format='plain',
                    timeout=7200,                       # 2 Ore
                    sleep_time=0.2,
                    verbose=False,                      # True
                    remove_section_headers=False,
                    skip_non_songs=True,
                    excluded_terms=None,
                    replace_default_terms=False,
                    retries=0)
    print("Genius Client created: ", genius)


    metadata_path = os.path.join(data_path, "raw/metadata_artists.csv")
    dataframe = pd.read_csv(metadata_path)                                          # It needs "openpyxl" to be installed
    pprint(dataframe.head())
    print("Dataframe Length: ", len(dataframe))
    dataframe_testi = pd.DataFrame(columns=["Artist", "Title", "Mood", "Song"])
    print("Length of Dataframe Testi: ", len(dataframe_testi))
    dataframe_testi.head()

    # Retrieving Lyrics
    HEADER_REMOVE = False
    pattern = r"^\n+|\n{2,}"
    filling_counter = 0
    with tqdm(range(len(dataframe))) as pbar:
        for entry, i in zip(dataframe.index, pbar):
            pbar.set_postfix(SEARCH=f" {dataframe['Artist'][entry]} - {dataframe['Title'][entry]}")
            curr_lyric = extract_song(dataframe["Artist"][entry], dataframe["Title"][entry], header_remove=HEADER_REMOVE)
            if curr_lyric != None:
                dataframe_testi.loc[filling_counter, 'Artist'] = dataframe["Artist"][entry]
                dataframe_testi.loc[filling_counter, 'Title'] = dataframe["Title"][entry]
                dataframe_testi.loc[filling_counter, 'Mood'] = dataframe['Mood'][i]
                dataframe_testi.loc[filling_counter, 'Song'] = re.sub(pattern, '', curr_lyric)
                filling_counter += 1
    print("Processo Completato")
    pprint(dataframe.head())