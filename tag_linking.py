import gensim
import time

model_path = './models/GoogleNews-vectors-negative300.bin'

model = None

def second(el):
    return el[1]

def load_model():
    print("Loading model... it can take some time...")
    return gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


def tag_to_set_similarity(tag, tag_set):
    return [(w, model.similarity(tag, w)) for w in tag_set]


def compute_tagset_similarity(in_tagset, tgt_tagset):
    similarities = {}
    for t in in_tagset:
        similarity_list = tag_to_set_similarity(t, tgt_tagset)
        similarity_list.sort(key=second, reverse=True)
        similarities[t] = similarity_list
    return similarities



in_set = set(['dog', 'painting', "country"])
tgt_set = set(['animal', 'cat', 'state', 'art', 'museum', 'horse', 'kingdom'])

model = load_model()

result = compute_tagset_similarity(in_set, tgt_set)
print("Dog:")
print(result['dog'])
print("=====================")
print("Country:")
print(result['country'])
print("=====================")
print("Painting:")
print(result['painting'])
