import sys
import os
import re
from nltk import pos_tag
from nltk import word_tokenize
from nltk import sent_tokenize

ann_in_directory = "./ann"
txt_in_directory = "./txt"
""" the text directory should contain the same files as the ann directory """
bio_out_directory = "./bio_orig"
bio_out_directory2 = "./bio_ken"


class Entity:
    # The class "constructor" - It's actually an initializer
    def __init__(self,eid,start_pos,end_pos,entity_text):
        self.eid = eid
        self.start_pos = int(start_pos)
        self.end_pos = int(end_pos)
        self.entity_text = entity_text


def norm_and_split_text(t):
    text = re.sub('[^a-zèéeêëėęûüùúūôöòóõœøîïíīįìàáâäæãåçćč&@#A-ZÇĆČÉÈÊËĒĘÛÜÙÚŪÔÖÒÓŒØŌÕÎÏÍĪĮÌ0-9_ \']', "*", t)
    wrds = text.split()
    return wrds


def write_bio_file_ken(file, text, start_end_positions):
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
                
                file.write(("\t".join((nltk_token, nltk_tag, iob_tag)) + "\n").encode("UTF-8"))
                
                if iob_tag == "B":
                    iob_tag = "I"
        else:
            if nltk_pos != 0:
                print("Something's fishy at %s (expected '%s' from '%s', but got '%s' from '%s')" % (text_pos, nltk_char, nltk_token, text_char, text[text_pos-5:text_pos + 5]))
                raise Exception()

            if text_char.isspace():
                pass
                # if text_pos < len(text):
                #     # eat whitespace sequence. annotation tool collapses mulitple whitespace
                #     # so we update the iob_offset with whitespace sequence length - 1.
                #     text_char = text[text_pos]
                #     while text_char.isspace() and text_pos + 1 < len(text):
                #         text_pos += 1
                #         text_char = text[text_pos]
                #         iob_offset += 1

            else:
                # eat non-whitespace characters and make a special token out of them
                non_nltk_token = text_char
                text_char = text[text_pos]
                while not text_char.isspace() and text_pos < len(text):
                    non_nltk_token += text_char
                    text_pos += 1
                    text_char = text[text_pos]

                if len(non_nltk_token) > 0:
                    result.append((non_nltk_token, non_nltk_token, iob_tag))
                    
                    file.write(("\t".join((non_nltk_token, non_nltk_token, iob_tag)) + "\n").encode("UTF-8"))

                    if iob_tag == "B":
                        iob_tag = "I"




def get_postags_from_nltk(text):
    sents = sent_tokenize(text)
    tokenlist_from_nltk = []
    pos_tag_list_from_nltk = []
    for sent in sents:
        #print (sent)
        pos_tagged_sent = pos_tag(word_tokenize(sent))
        #print(pos_tagged_sent)
        for word_with_pos in pos_tagged_sent:
            word = word_with_pos[0]
            partofspeech = word_with_pos[1]
            tokenlist_from_nltk.append(word)
            pos_tag_list_from_nltk.append(partofspeech)
            #print (word_with_pos[0],word_with_pos[1])
    #alltokens = norm_and_split_text(text)

    #normalized_text = ' '.join(alltokens)
    text_after_nltk = ' '.join(tokenlist_from_nltk)

    #print(len(text),len(text_after_nltk),len(normalized_text))
    text = re.sub('\s\s+',' ',text)
    # print (text)
    # print(text_after_nltk)

    corrected_text_after_nltk = ''
    corrected_pos_tag_list_from_nltk = []
    j=0
    for i in range(0,len(text)-1):
        if j >= len(text_after_nltk) or text_after_nltk[j] == ' ':
            partofspeech = pos_tag_list_from_nltk.pop(0)
            word = tokenlist_from_nltk.pop(0)
            #print (word,partofspeech)

            if text[i] == ' ':
                corrected_pos_tag_list_from_nltk.append(partofspeech)
                #print ("corrected:",word,"appended:",partofspeech)

            else:
                j += 1
        #else:
            #print(text[i],text_after_nltk[j])
        corrected_text_after_nltk += text_after_nltk[j]

        j+=1


    print (text)
    print (corrected_text_after_nltk)


    #print (text)
    #print (normalized_text)
    return corrected_pos_tag_list_from_nltk


if __name__ == "__main__":

    """
    READ .ann FILES
    """
    entities_per_file = dict()
    for filename in os.listdir(ann_in_directory):
        if ".ann" in filename:
            print (filename)
            with open(ann_in_directory+"/"+filename,'r', encoding='utf-8') as ann_file:
                entities = []
                for line in ann_file:
                    (T_id,type_start_end,entity_text) = line.rstrip().split("\t")
                    (entity_type,start_pos,end_pos) = type_start_end.split(" ")
                    entity = Entity(T_id,start_pos,end_pos,entity_text)
                    entities.append(entity)
                entities_per_file[filename.replace('.ann','')] = entities
            ann_file.close()

    """
    READ CORRESPONDING .txt FILES
    """
    for filename in os.listdir(txt_in_directory):
        if ".txt" in filename:
            if filename.replace('.txt','') not in entities_per_file:
                print("ERROR: Text file has no equivalent in ann directory:",filename)
                break

            print(filename)
            entities = entities_per_file[filename.replace('.txt','')]


            text = ''
            with open(txt_in_directory+"/"+filename,'r', encoding='utf-8') as txt_file:
                text = txt_file.read()


            print(len(text))
            text = re.sub('\r\n','\n',text)
            text = re.sub('\s\s+',' ',text)
            print(len(text))

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
                        

                # if best_found != entity.start_pos:
                #     print("Discrepancy for %s. Found at %s, expected %s, delta %s " % (entity.eid, best_found, entity.start_pos, shortest_distance))
                if best_found <= 0:
                    print("Discrepancy in start pos. Found at %s, expected %s, delta %s " % (best_found, entity.start_pos, shortest_distance))

                start_end_positions[best_found] = best_found + (entity.end_pos - entity.start_pos)
                

            with open(bio_out_directory+"/"+filename.replace('txt','bio'),'wb') as out_bio:
                write_bio_file_ken(out_bio, text, start_end_positions)


