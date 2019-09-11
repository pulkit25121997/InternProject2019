# Importing the libraries for Data-Processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stanfordnlp 

# Setting path for the StanfordNLP to take the Hindi-Models.
config = {
	'processors': 'tokenize,mwt,pos,lemma, ner', # Comma-separated list of processors to use
	'lang': 'hi', # Language code for the language to build the Pipeline in
	'tokenize_model_path': '/home/pulkit/stanfordnlp_resources/hi_hdtb_models/hi_hdtb_tokenizer.pt', # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
	'mwt_model_path': '/home/pulkit/stanfordnlp_resources/hi_hdtb_models/hi_hdtb_mwt_expander.pt',
	'pos_model_path': '/home/pulkit/stanfordnlp_resources/hi_hdtb_models/hi_hdtb_tagger.pt',
	'pos_pretrain_path': '/home/pulkit/stanfordnlp_resources/hi_hdtb_models/hi_hdtb.pretrain.pt',
	'lemma_model_path': '/home/pulkit/stanfordnlp_resources/hi_hdtb_models/hi_hdtb_lemmatizer.pt',
	'depparse_model_path': './hi_hdtb_models/hi_hdtb_parser.pt',
	'depparse_pretrain_path': './hi_hdtb_models/hi_hdtb.pretrain.pt'
}

# Making the StanfordNLP Pipeline
nlp = stanfordnlp.Pipeline(**config)

# Defining the POS Tagging for StanfordNLP
pos_dict = {
'CC': 'coordinating conjunction','CD': 'cardinal digit','DT': 'determiner',
'EX': 'existential there (like: \"there is\" ... think of it like \"there exists\")',
'FW': 'foreign word','IN':  'preposition/subordinating conjunction','JJ': 'adjective \'big\'',
'JJR': 'adjective, comparative \'bigger\'','JJS': 'adjective, superlative \'biggest\'',
'LS': 'list marker 1)','MD': 'modal could, will','NN': 'noun, singular \'desk\'',
'NNS': 'noun plural \'desks\'','NNP': 'proper noun, singular \'Harrison\'',
'NNPS': 'proper noun, plural \'Americans\'','PDT': 'predeterminer \'all the kids\'',
'POS': 'possessive ending parent\'s','PRP': 'personal pronoun I, he, she',
'PRP$': 'possessive pronoun my, his, hers','RB': 'adverb very, silently,',
'RBR': 'adverb, comparative better','RBS': 'adverb, superlative best',
'RP': 'particle give up','TO': 'to go \'to\' the store.','UH': 'interjection errrrrrrrm',
'VB': 'verb, base form take','VBD': 'verb, past tense took',
'VBG': 'verb, gerund/present participle taking','VBN': 'verb, past participle taken',
'VBP': 'verb, sing. present, non-3d take','VBZ': 'verb, 3rd person sing. present takes',
'WDT': 'wh-determiner which','WP': 'wh-pronoun who, what','WP$': 'possessive wh-pronoun whose',
'WRB': 'wh-abverb where, when','QF' : 'quantifier, bahut, thoda, kam (Hindi)','VM' : 'main verb',
'PSP' : 'postposition, common in indian langs','DEM' : 'demonstrative, common in indian langs'
}

# Data provided by Sovan Sir
data = pd.read_csv('Pulkit.csv')

# Make sentences out of given word corpus 
def make_sentences(data):
	sentences = []
	sentence = data['word'][0]
	for i in range(1, len(data['word'])):
		if(data['index'][i] != data['index'][i-1]+1):
			sentences.append(sentence)
			sentence = str(data['word'][i])
		else:
			sentence = sentence + " " + str(data['word'][i])
	sentences.append(sentence)
	return sentences


# Function for extracting the POS tag from the sentence after pipelining
def extract_pos(doc):
    parsed_text = {'word':[], 'pos':[], 'exp':[]}
    for sent in doc.sentences:
        for wrd in sent.words:
            if wrd.pos in pos_dict.keys():
                pos_exp = pos_dict[wrd.pos]
            else:
                pos_exp = 'NA'
            parsed_text['word'].append(wrd.text)
            parsed_text['pos'].append(wrd.pos)
            parsed_text['exp'].append(pos_exp)
    return pd.DataFrame(parsed_text)



sentences = make_sentences(data)
#print(sentences)


# Create a new dataframe to store the results
new_data = pd.DataFrame({"word":[], "pos":[], "exp": []}) 

for sentence in sentences:
	#print(sentence)
	try:
		hindi_doc = nlp(sentence)
		new_data = new_data.append(extract_pos_ner(hindi_doc))
		#print(new_data)
	except:
		print("I am in exception")
		#new_data = new_data.append("", "", "" )

# Saving the results into a CSV file
new_data.to_csv('/home/pulkit/InternProject/Testing.csv')