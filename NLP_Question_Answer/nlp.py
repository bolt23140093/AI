import jieba
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm 
def file_read(file,mode):
    try:
        with open(file,mode,encoding="utf8") as f: 
            text=f.read()
            return text
    except IOError:
        print('Error: cannot open file')
        
def sentence_list(text):
    text = BeautifulSoup(text, 'html5lib').get_text()#移除html標籤
    sg_list = jieba.cut(text)
    return list(sg_list)

def tokenize(text):
    token_free_words = [word for word in text if word not in "?。」「，.!/;:\n\'\"、@#%^&*"]
    return token_free_words

stop_words = file_read('stop_words.txt','r')

def remove_noise(tokens):
    noise_free_words = [word for word in tokens if word not in stop_words]
    return noise_free_words


def clean_text(text):
    word_list = sentence_list(text)#轉成單字list
    word_list = tokenize(word_list)#去標點
    word_list = remove_noise(word_list)#去noise
    return word_list


##why enclosing function ?
##http://blog.ittraining.com.tw/2019/12/python-why-enclosing-function.html
def set_word_vector(word_vecs,dim):
    def word_feature(text,wv=word_vecs,dim=dim):
        emb_cnt = 0
        avg_emb = np.zeros((dim,))

        for word in clean_text(text):

            if word in wv:
                #print(word) 
                avg_emb += wv[word]
                emb_cnt += 1
        avg_emb /= emb_cnt
        return avg_emb
    return word_feature

def euclidean_distance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))
	
# 開啟詞向量檔案
def load_WordVector():
    dim = 0
    word_vecs= {}
    with open('cna.cbow.cwe_p.tar_g.512d.0.txt',encoding="utf-8") as f:
        for line in tqdm(f):
        # 詞向量有512維,由word以及向量中的元素共513個
        # 以空格分隔組成詞向量檔案中一行
            tokens = line.strip().split()

            # txt中的第一列是兩個整數，分別代表有多少個詞以及詞向量維度
            if len(tokens) == 2:
                dim = int(tokens[1])
                continue
            #詞向量從第2列開始
            word = tokens[0] 
            vec = np.array([ float(t) for t in tokens[1:] ])
            word_vecs[word] = vec
			
    
    return dim,word_vecs