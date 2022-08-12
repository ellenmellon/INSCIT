import os
import json, fileinput
import csv
import pathlib
from argparse import ArgumentParser


# Proccesses one of the files
def process_file(user_filename, output_file):

    out_dir = os.path.dirname(output_file)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    with open(user_filename, 'rb') as source, open(output_file, 'a') as dest:
        # Load the json objects from file and then open the new tsv
        # file
        tsv_writer = csv.writer(dest, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(['id', 'text', 'title']) # set headers
        json_reader = json.load(source)

        # Move the data from json format to tsv
        for json_object in json_reader:
            current_title = json_object['title'].strip()

            # Parsing the text from each json object in the file
            for pid, passage in enumerate(json_object['passages']):
                new_row = []
                cur_id = passage['id']
                new_row.append(cur_id)
                new_row.append(' '.join(passage['text'].split()))
                
                # Use all subtitles
                new_row.append(" [SEP] ".join(passage['titles']))
                tsv_writer.writerow(new_row)


def main(raw_corpus_dir, output_file):
    for filename in os.listdir(raw_corpus_dir):
        print('processing', filename)
        dir_or_file = os.path.join(raw_corpus_dir, filename)
        if os.path.isfile(dir_or_file):
            process_file(dir_or_file, output_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_corpus_dir", type=str, default='../../data/text_0420_processed')
    parser.add_argument("--output_file", type=str, default='./retrieval_data/wikipedia/full_wiki_segments.tsv')
    args = parser.parse_args()
    main(args.raw_corpus_dir, args.output_file)
