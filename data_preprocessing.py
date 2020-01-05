from __future__ import unicode_literals

import os
import re
import csv
import shutil
import requests
import subprocess
from io import BytesIO
from zipfile import ZipFile

import youtube_dl
import numpy as np
import pandas as pd
from pydub import AudioSegment

from utils import create_directory_if_not_exist, config_local

TRAIN_DEBATES = [
    '1st_presidential',
    '2nd_presidential',
    'vice_presidential'
]

TEST_DEBATES = [
    '3rd_presidential',
    '9th_democratic',
    'trump_acceptance',
    'trump_at_wef',
    'trump_address_to_congress',
    'trump_at_tax_reform_event',
    'trump_miami_speech'
]

def main():
    create_directory_if_not_exist('data/debates')

    audio_data = pd.read_csv('data/claim_audio_intervals.csv')

    for debate_group, debate_data in audio_data.groupby(['DEBATE', 'AUDIO_URL']):
        debate, audio_url = debate_group
        timestamps_df = debate_data[['CLAIM_ID','START', 'END']]

        debate_dir = os.path.join('data', 'debates', debate)
        create_directory_if_not_exist(debate_dir)

        audio_filename = 'audio.wav'

        download_audio(debate, debate_dir, audio_filename, audio_url)
        split_audio(debate, debate_dir, audio_filename, timestamps_df)
        os.remove(os.path.join(debate_dir, audio_filename))
        opensmile_compare_features(debate, debate_dir)
        shutil.rmtree(os.path.join(debate_dir, 'audio_segments'))

    download_train_transcriptions()
    download_test_transcriptions()

def download_audio(debate, debate_dir, audio_filename, audio_url):
    audio_file = os.path.join(debate_dir, audio_filename)

    if os.path.exists(audio_file):
        print('Audio file exists.')
    else:
        with youtube_dl.YoutubeDL({ 'format': 'bestaudio', 'outtmpl': audio_file }) as ydl:
            ydl.download([audio_url])

def split_audio(debate, debate_dir, audio_filename, timestamps_df):
    audio_segments_dir = os.path.join(debate_dir, 'audio_segments')
    create_directory_if_not_exist(audio_segments_dir)

    audio = AudioSegment.from_file(os.path.join(debate_dir, audio_filename))

    for index, row in timestamps_df.iterrows():
        chunk_data = audio[float(row['START']) * 1000:float(row['END']) * 1000]
        chunk_data.export(os.path.join(debate_dir, 'audio_segments', 'audio_%s_%s.wav' % (debate, int(row['CLAIM_ID']))), format='wav')

def opensmile_compare_features(debate, debate_dir):
    audio_segments_dir = os.path.join(debate_dir, 'audio_segments')
    audio_features_dir = os.path.join(debate_dir, 'audio_features')

    create_directory_if_not_exist(audio_features_dir)

    for audio_segment in sorted(os.listdir(audio_segments_dir)):
        if not audio_segment.endswith('.wav'): continue
        audio_name = audio_segment[:-4]

        opensmile_dir = config_local().get('opensmile')
        subprocess.run([
            os.path.join(opensmile_dir, 'inst/bin/SMILExtract'),
            '-C',
            os.path.join(opensmile_dir, 'config/IS13_ComParE.conf'),
            '-I',
            os.path.join(audio_segments_dir, audio_segment),
            '-O',
            os.path.join(audio_features_dir, audio_name + '.arff'),
            '-instname',
            audio_name
        ])

def download_train_transcriptions():
    with open('data/debates/1st_presidential/transcription.txt', 'wb') as transcription: transcription.write(requests.get('https://raw.githubusercontent.com/clef2018-factchecking/clef2018-factchecking/master/data/task2/English/Task2-English-1st-Presidential.txt').content)
    with open('data/debates/2nd_presidential/transcription.txt', 'wb') as transcription: transcription.write(requests.get('https://raw.githubusercontent.com/clef2018-factchecking/clef2018-factchecking/master/data/task2/English/Task2-English-2nd-Presidential.txt').content)
    with open('data/debates/vice_presidential/transcription.txt', 'wb') as transcription: transcription.write(requests.get('https://raw.githubusercontent.com/clef2018-factchecking/clef2018-factchecking/master/data/task2/English/Task2-English-Vice-Presidential.txt').content)

def download_test_transcriptions():
    test_debates_mapping = {
        'task2-en-file1.txt': '3rd_presidential',
        'task2-en-file2.txt': '9th_democratic',
        'task2-en-file3.txt': 'trump_miami_speech',
        'task2-en-file4.txt': 'trump_at_tax_reform_event',
        'task2-en-file5.txt': 'trump_at_wef',
        'task2-en-file6.txt': 'trump_acceptance',
        'task2-en-file7.txt': 'trump_address_to_congress',
    }
    test_transcriptions_path = 'clef18_fact_checking_lab_submissions_and_scores_and_combinations/task2_gold/English/'
    zipfile = requests.get('http://alt.qcri.org/clef2018-factcheck/data/uploads/clef18_fact_checking_lab_submissions_and_scores_and_combinations.zip', stream=True).content

    zipfile_obj = ZipFile(BytesIO(zipfile))
    for key, debate in test_debates_mapping.items():
        transcription = zipfile_obj.getinfo(os.path.join(test_transcriptions_path, key))
        transcription.filename = 'transcription.txt'
        zipfile_obj.extract(transcription, 'data/debates/' + debate)

if __name__ == '__main__':
    main()
