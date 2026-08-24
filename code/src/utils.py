import re
import yaml
import random
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

# Lyrics Retriever Genius API
import urllib.request
from urllib.error import HTTPError
# from lyricsgenius import Genius


# Read configuration file
def read_config_file(config_path:str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg



# Retrieve Song from Genius
""" This function extracts the lyrics of a song by the specified artist and title using the Genius API.
    Parameters:
        - artist (str): The name of the artist.
        - title (str): The title of the song.
        - header_remove (bool, optional): Whether to remove section headers (e.g., Chorus, etc.) from the lyrics.
    Returns:
        - str: The lyrics of the song.
    Note:
        - If the song is not found or an HTTP error occurs, it returns None and prints an error message.
"""
def extract_song(artist:str, title:str, header_remove:bool=True) -> str :
    try:
        song = genius.search_song(title=str(title), artist=str(artist), get_full_info=False)
    except HTTPError:
        song = None
    
    if song != None:
        lyric = genius.lyrics(song_url=song.url, remove_section_headers=header_remove) # Remove Section Headers (e.g Verse1, Chrous, ecc)
        splitting_string = song.title + " " + "Lyrics"
        lyric = lyric.split(splitting_string)
        if len(lyric) > 1: lyric = re.split("[0-9]*Embed{1}$", lyric[1])
        else: lyric = re.split("[0-9]*Embed{1}$", lyric[0])
        return lyric[0]
    else:
        print(f"{artist} - {title} -> Nothing Found")
        return None




# Data Construction utilities
""" Constructs data from a specified CSV file containing song lyrics.
    Parameters:
        - data_file (str, optional): The path to the CSV file containing song lyrics. If not provided, it will print an error message.
    Returns:
        - tuple: A tuple containing two lists:
            - text_lines (list): A list of individual lines from the song lyrics.
            - text_lines_pairs (list): A list of pairs of consecutive lines from the song lyrics.
    Notes:
        - If 'data_file' is not provided, or if it does not have a '.csv' extension, an error message is printed.
        - The function reads the CSV file, splits the song lyrics into individual lines, and constructs pairs of consecutive lines.
        - It also handles cases where empty lines are encountered in the song lyrics.
"""
def construct_data(data_file:str=None):
    text_lines = []
    text_lines_pairs = []
    if data_file == None: print("Error: You didn't provide a right path to a data file")
    if data_file == None and ".cvs" not in data_file: print("Error: The datafile you provided isn't not .csv file")
    else:
        df = pd.read_csv(data_file)
        with tqdm(df.index, desc="Data Costruction") as pbar:
            for _, i in zip(pbar, df.index):
                pieces_of_txt = re.split(r"\n", df["Song"][i])
                if "" in pieces_of_txt: pbar.set_postfix({"SAMPLE":i, "MSG": "Empty space encountered"}); pbar.update(); pieces_of_txt.remove("")
                for j in range(len(pieces_of_txt)-1):                       # Since the last verse of the song has no successor
                    if (pieces_of_txt[j], pieces_of_txt[j+1]) not in text_lines_pairs:
                        text_lines_pairs.append((pieces_of_txt[j], pieces_of_txt[j+1]))         # (current_line, next_line) pairs construction
                    if pieces_of_txt[j] not in text_lines:
                        text_lines.append(pieces_of_txt[j])                                     # current lines list construction
                if pieces_of_txt[len(pieces_of_txt)-1] not in text_lines:   # Since we want to feed into the line list also the last one with no successors
                    text_lines.append(pieces_of_txt[len(pieces_of_txt)-1])
                pbar.set_postfix(SAMPLE=i)
                pbar.update()
    
    return text_lines, text_lines_pairs

def slice_txt_data(data:list, n_slice:int=2):
    if n_slice > 3: print("Error: Maximum slice is 3 (Training, Validation, Test)")
    else:
        length_slice_i = len(data)//n_slice
        if n_slice == 2:
            train_data = data[:length_slice_i]
            valid_data = data[length_slice_i:]

            return train_data, valid_data
        if n_slice == 3:
            train_data = data[:length_slice_i]
            valid_data = data[length_slice_i:2*length_slice_i]
            test_data = data[2*length_slice_i:]

            return train_data, valid_data, test_data

def sample_data(txt_lines:List, txt_lines_pair:List[tuple], n_samples:int=None, n_alternatives:int=200) -> (Dict, Dict):
    in_data = {}
    ground_truth = {}
    
    if n_samples == None: correct_candidates = random.sample(txt_lines_pair, len(txt_lines_pair))
    else: correct_candidates = random.sample(txt_lines_pair, n_samples)

    with tqdm(range(len(correct_candidates)), desc="Input/Ground Truth") as pbar:
        try:
            for i, (first_line, second_line) in zip(pbar, correct_candidates):
                ground_truth[first_line] = second_line                      # Ground Truth List of (sentence_1, sentence_2) pairs  
                in_data.setdefault(first_line,[]).append(second_line)       # Input Data: Here we add the correct (sentence_1, sentence_2) pair
        
                random_alternatives = random.sample(txt_lines, n_alternatives)

                in_data[first_line].extend([alt for alt in random_alternatives])
                pbar.set_postfix(SAMPLE=i)
                pbar.update()
        except:
            print("ERRORE: ", correct_candidates[i])
    return in_data, ground_truth



# Save/Open data
""" Saves data to a specified file based on the file content type.
    Parameters:
        - data (list): The data to be saved, either a list of lines or a list of line pairs.
        - type (str): The type of data being saved, either "lines" or "lines-pairs".
        - path (str): The path to the file where the data will be saved.
    Notes:
        - If 'type' or 'path' is not provided, an error message is printed.
        - The function checks the 'type' parameter to determine how to format and save the data:
        - If 'type' is "lines," each item in the 'data' list is written to the file as a separate line.
        - If 'type' is "lines-pairs," each item in the 'data' list is a pair of lines, and they are written to the file with a delimiter ('>|<') between them.
"""
def save_data(data, type:str, path:str):
    if type == None or path == None: print("Error: No path for saving provided")
    else:
        if type == "lines":
            with open(path, 'w') as file:
                for line in data: file.write(line+"\n")
        if type == "lines-pairs":
            with open(path, 'w') as file:
                for line_pair in data: file.write(line_pair[0]+">|<"+line_pair[1]+"\n")

""" Opens and reads data from a specified file based on the file content type.
    Parameters:
        - path (str): The path to the file containing the data.
        - type (str): The type of data to be read, either "lines" or "lines-pairs."
    Returns:
        - list: A list of lines or line pairs depending on the specified 'type.'
    Notes:
        - If 'type' or 'path' is not provided, an error message is printed.
        - The function checks the 'type' parameter to determine how to read and format the data:
        - If 'type' is "lines," it reads the file and returns a list of lines.
        - If 'type' is "lines-pairs," it reads the file and returns a list of pairs of lines, splitting them using the delimiter ('>|<').
"""
def open_data(path:str, type:str):
    if type == None or path == None: print("Error: No path to valid files provided")
    else:
        if type == "lines":
            with open(path, 'r') as file:
                lines = file.readlines()
                lines = [re.sub("\n", "", line) for line in lines]
                return lines
        if type == "lines-pairs":
            with open(path, 'r') as file:
                lines = []
                for line in file.readlines():
                    line = re.sub("\n", "", line)
                    l = re.split(r"[>|<]+", line)
                    lines.append(tuple(l))
            return lines