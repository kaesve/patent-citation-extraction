import sys
import os
import re
from nltk import pos_tag
from nltk import word_tokenize
from nltk import sent_tokenize


class Entity:
    # The class "constructor" - It's actually an initializer
    def __init__(self,eid,start_pos,end_pos,entity_text):
        self.eid = eid
        self.start_pos = int(start_pos)
        self.end_pos = int(end_pos)
        self.entity_text = entity_text

def filter_files(ext, ls):
    return [ name[:-len(ext)] for name in ls if name[-len(ext):] == ext ]


def process_patent(text, start_end_positions = dict()):
    iob_points = []

    for start in sorted(start_end_positions):
        iob_points.append(start)
        iob_points.append(start_end_positions[start])
    

    sents = sent_tokenize(text)
    nltk_tagged_tokens = []
    for sent in sents:
        nltk_tagged_tokens.extend(pos_tag(word_tokenize(sent)))

    nltk_index = 0
    nltk_pos = 0
    text_pos = 0

    iob_index = 0
    iob_offset = 0
    iob_tag = "O"

    result = []

    while text_pos < len(text):
        if nltk_index < len(nltk_tagged_tokens):
            nltk_token, nltk_tag = nltk_tagged_tokens[nltk_index]

        if iob_index < len(iob_points):
            iob_boundary = iob_points[iob_index]
            if iob_boundary == text_pos + iob_offset:
                iob_index += 1
                if iob_tag == "O":
                    iob_tag = "B"
                else:
                    iob_tag = "O"

        text_char = text[text_pos]
        nltk_char = nltk_token[nltk_pos]
        text_pos += 1

        if nltk_index < len(nltk_tagged_tokens) and text_char == nltk_char:
            nltk_pos += 1
            if nltk_pos >= len(nltk_token):
                nltk_pos = 0
                nltk_index = nltk_index + 1
                result.append((nltk_token, nltk_tag, iob_tag))
                
                if iob_tag == "B":
                    iob_tag = "I"
        else:
            if nltk_pos != 0:
                print("Something's fishy at %s (expected '%s' from '%s', but got '%s' from '%s')" % (text_pos, nltk_char, nltk_token, text_char, text[text_pos-5:text_pos + 5]))
                raise Exception()

            if text_char.isspace():
                pass
            else:
                # eat non-whitespace characters and make a special token out of them
                non_nltk_token = text_char
                while text_pos < len(text) and not text[text_pos].isspace():
                    non_nltk_token += text[text_pos]
                    text_pos += 1

                if len(non_nltk_token) > 0:
                    result.append((non_nltk_token, non_nltk_token, iob_tag))

                    if iob_tag == "B":
                        iob_tag = "I"
    return result


if __name__ == "__main__":

    # TODO: It would be nicer to make these command line arguments.
    ann_in_directory = "./ann"
    txt_in_directory = "./txt"
    """ the text directory should contain the same files as the ann directory """
    bio_out_directory = "./bio_test"


    entity_files = set(filter_files(".ann", os.listdir(ann_in_directory)))

    for filename in filter_files(".txt", os.listdir(txt_in_directory)):
        if filename not in entity_files:
            print("ERROR: Text file has no equivalent in ann directory:",filename)
            continue

        print(filename)
        
        # Parse entities
        with open(f"{ann_in_directory}/{filename}.ann", 'r', encoding='utf-8') as ann_file:
            entities = []
            for line in ann_file:
                T_id, type_start_end, entity_text = line.rstrip().split("\t")
                entity_type, start_pos, end_pos = type_start_end.split(" ")
                entities.append(Entity(T_id, start_pos, end_pos, entity_text))


        with open(f"{txt_in_directory}/{filename}.txt", 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()

        text = re.sub('\r\n','\n', text)
        text = re.sub('\s\s+',' ', text)

        # Realign annotations -- the annotation tool used slightly different versions of the text
        # files, so the ranges from the annotation file do not align with our text files.
        #
        # We use the full strings provided by the annotation file, to find the exact match in our
        # file that is closest to the corresponding text range for that annotation. This isn't
        # perfect, but works in most cases. The remaining cases will be logged and can be cleaned 
        # up by hand if desired.
        start_end_positions = dict()
        for entity in entities:    
            shortest_distance = len(text)
            best_found = 0
            found = 0
            while found != -1:
                found = text.find(entity.entity_text.strip(), found + 1)

                delta = abs(found - entity.start_pos)
                if delta < shortest_distance:
                    shortest_distance = delta
                    best_found = found
                    
            if best_found <= 0:
                print("Discrepancy in start pos. Found at %s, expected %s, delta %s " % (best_found, entity.start_pos, shortest_distance))

            start_end_positions[best_found] = best_found + (entity.end_pos - entity.start_pos)
            

        # Preprocess patent
        processed = process_patent(text, start_end_positions)
        with open(f"{bio_out_directory}/{filename}.bio", 'wb') as out_bio:
            for token in processed:
                line = "\t".join(token) + "\n"
                out_bio.write(line.encode("UTF-8"))


