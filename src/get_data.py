from tqdm import tqdm
import editdistance
import logging
logging.basicConfig(level=logging.INFO)
# conceptnet version 5.8

def extract_conceptnet_triples(lang, input_file, stop_words):
    with open(input_file) as f:
        conceptnet_data = f.readlines()

    all_triple = dict()
    relation_dict = dict()
    for node_edge in tqdm(conceptnet_data, ncols=100, desc="extract triples from conceptnet"):
        data_ = node_edge.split("\t")
        relation = data_[1][3:] # /r/relation
        start = data_[2][6:]  # /c/ab/start of edge
        end = data_[3][6:]  # /c/ab/end of edge
        if (data_[2][3:5] == lang) & (data_[3][3:5] == lang):
            start = clean_triple(start)
            relation = clean_triple(relation)
            end = clean_triple(end)
            if start in stop_words or end in stop_words:
                continue

            if start not in all_triple.keys():
                all_triple[start] = list()
            if start != end:
                all_triple[start].append((relation, end))
            if relation not in relation_dict.keys():
                relation_dict[relation] = len(relation_dict)
    return all_triple, relation_dict




def clean_triple(node_edge):
    node_ = node_edge.split("/")[0]
    if node_ == "dbpedia":
        node_ = node_edge.split("/")[1]
    clean_node_edge = node_.replace("_", " ").lower()
    return clean_node_edge

def confirm_triple_in_text(text, triple_dict, ngrams=6):
    text_with_concept = list()
    for line in tqdm(text, ncols=100, desc="find concept in text"):
        line = line.split()
        if len(line) <= 1:
            continue
        line_ = list()
        if ngrams == 1:
            line_=line
        else:
            for n in range(1, ngrams + 1):
                line_.extend([" ".join(line[i:i + n]) for i in range(len(line))])
            line_ = list(set(line_))
        for ngram in line_:
            if ngram in triple_dict:
                text_with_concept.append(line)
                break

    return text_with_concept





def find_max_len(concept_dict):
    max_len = 0
    for concept in tqdm(concept_dict.keys(), ncols=100, desc="find max length in concept"):
        concept_ = concept.split()
        max_len = max(max_len, len(concept_))
    print("max length of concept {}".format(max_len))
    return max_len

def clean_wiki(input_file):
    prepro_txt = list()
    with open(input_file) as f:
        raw_txt = f.readlines()
        raw_txt = [txt.split("\n")[0] for txt in raw_txt]
    for line in tqdm(raw_txt, ncols=100, desc="clean wiki text"):
        if len(line) <= 1:
            continue
        else:
            line = line.lower()
            prepro_txt.append(line)
    return prepro_txt

def prepro_wiki(file_path, tokenizer=None, ngram=6):
    lines = torch.load(file_path)
    prepro_ = list()
    for line in tqdm(lines, desc="make word dict ", ncols=100):
        #line = tokenizer(line)
        #line = [token.text for token in line]
        line = line.split()
        line = make_ngram(line, ngram=6)
        prepro_.append(line)
    return prepro_

def make_ngram(line, ngram=6):
    line_ = list()
    ln = min(len(line), ngram)
    for n in range(1, ln + 1):
        line_.extend([" ".join(line[i:i + n]) for i in range(len(line))])
    line_ = list(set(line_))
    return line_


def filter_commonsense_triple(all_triple, relation_filter):
    commonsense_triple = dict()
    for s in tqdm(all_triple.keys(), ncols=100, desc="filter commonsense triple from all triple"):
        r_o = all_triple[s]
        for r, o in r_o:
            if r in relation_filter:
                if s not in commonsense_triple.keys():
                    commonsense_triple[s] = list()
                commonsense_triple[s].append((r,o))
    return commonsense_triple

def make_concept_dict(commonsense_triple):
    concept_dict = dict() # 1165286
    for s, r, o in tqdm(commonsense_triple, ncols=100, desc="make concept dictionary"):
        if s not in concept_dict:
            concept_dict[s] = len(concept_dict)
        if o not in concept_dict:
            concept_dict[o] = len(concept_dict)
    return concept_dict


def split_wiki(wiki_data=None, split_n=20, path=None):
    leng = len(wiki_data) //split_n
    for i in tqdm(range(split_n), desc="split data to {} pieces".format(split_n), ncols=100):
        split_data = wiki_data[leng*i:leng*(i+1)]
        torch.save(split_data, path[i])

    logging.info("finished ! ")
    return None


def find_two_concept_in_line(data, common_triple):
    data = torch.load(data)
    triple_freq1 = list()
    triple_freq2 = list()
    for line in tqdm(data, desc="find two concept in line", ncols=100):
        find_start = set(line) & set(common_triple.keys())
        if len(find_start):  # start 있으면
            for start in find_start:
                ends = [e for r, e in common_triple[start]]
                relations = [r for r, e in common_triple[start]]
                find_end = set(line) & set(ends)
                if len(find_end):
                    for end in find_end:
                        idx = ends.index(end)
                        relation = relations[idx]
                        triple_freq2.append((start, relation, end))
                else:
                    for end in ends:
                        idx = ends.index(end)
                        relation = relations[idx]
                        triple_freq1.append((start, relation, end))

    return triple_freq1, triple_freq2



if __name__ =="__main__":
    import torch
    import spacy
    from tqdm import tqdm
    import os, sys

    split_num = 100
    ###############data split & prepro ###################################################################################
    #tokenizer = spacy.load("en_core_web_sm")
    # prepro_txt = clean_wiki("./data/wiki_dump_extracted_2021_sentence_per_line_enter_seperated_docs.txt")

    # split_file_path = "./data/split"
    #split_prepro_file_path = "./data/prepro_split"

    # if not os.path.exists(split_file_path):
    #     os.mkdir(split_file_path)
    # if not os.path.exists(split_prepro_file_path):
    #     os.mkdir(split_prepro_file_path)


    # file_list = list()

    #
    # file_list = [os.path.join(split_file_path,"file"+str(i)+"_.pkl") for i in range(split_num)]
    # split_wiki(wiki_data=prepro_txt,split_n=split_num ,path=file_list)
    # del prepro_txt

    # for i in tqdm(range(split_num), ncols=100, desc="split data extend to ngrams"):
    #     raw_path = os.path.join(split_file_path, "file"+str(i)+"_.pkl")
    #     prepro_path= os.path.join(split_prepro_file_path, "file"+str(i)+"_.pkl")
    #     prepro_data = prepro_wiki(raw_path)
    #     torch.save(prepro_data, prepro_path)

    #
    # conceptnet_raw_path = "/data/user15/workspace/rb_pjt/data/conceptnet-assertions-5.7.0.csv"
    # stop_words = ["a", "an", "am", "are","and", "be", "been", "being", "to","the","of", "is", "which"]
    # all_triple, relation_dict  = extract_conceptnet_triples("en",conceptnet_raw_path, stop_words)
    # torch.save(all_triple,"./data/all_triple.pkl")

    ################# find triplet frequency ###########################################################################
    split_prepro_file_path = "./data/prepro_split"
    split_freq_path_1 = "./data/freq_1"
    split_freq_path_2 = "./data/freq_2"

    if not os.path.exists(split_freq_path_1):
        os.mkdir(split_freq_path_1)
    if not os.path.exists(split_freq_path_2):
        os.mkdir(split_freq_path_2)


    file_list = [os.path.join(split_prepro_file_path, "file" + str(i) + "_.pkl") for i in range(split_num)]

    all_triple = torch.load("./data/all_triple.pkl")
    relation_filter = ["atlocation", "capableof", "causes", "causesdesire", "createdby", "desires", "hasa",
                       "hasfirstsubevent", "haslastsubevent", "hasprerequisite",
                       "hasproperty", "hassubevent", "locatednear", "madeof", "motivatedbygoal", "notcapableof",
                       "notdesires", "nothasproperty", "partof", "receivesaction", "usedfor"]

    # atomic_filter = ["atlocation", "capableof", "causes", "causesdesire",  "desires", "hasa",
    #                    "hasfirstsubevent", "haslastsubevent", "hasprerequisite",
    #                    "hasproperty", "hassubevent", "madeof", "motivatedbygoal",
    #                    "notdesires", "partof", "receivesaction", "usedfor"]

    common_triple = filter_commonsense_triple(all_triple, relation_filter)
    # atomic_triple = filter_commonsense_triple(all_triple, atomic_filter)
    del all_triple
    for i in tqdm(range(split_num), ncols=100, desc="find triplet in sentence"):
        raw_path = os.path.join(split_prepro_file_path, "file"+str(i)+"_.pkl")
        path_1 = os.path.join(split_freq_path_1, "file"+str(i)+"_.pkl")
        path_2 = os.path.join(split_freq_path_2, "file"+str(i)+"_.pkl")

        freq_1, freq_2 = find_two_concept_in_line(raw_path, common_triple)
        print("freq_1 {}".format(len(freq_1)))
        print("freq_2 {}".format(len(freq_2)))
        torch.save(freq_1, path_1)
        torch.save(freq_2, path_2)
        del freq_1
        del freq_2
#
# freq_1 = torch.load("/data/user15/workspace/rb_pjt/data/freq_1/file0_.pkl")
# freq_2 = torch.load("/data/user15/workspace/rb_pjt/data/freq_2/file0_.pkl")










