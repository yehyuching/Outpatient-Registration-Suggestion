
# coding: utf-8

# In[ ]:



import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

#compile sym_talk dictionary
talk_dic ={}
f_talk = open('symptom_talks.txt', encoding = 'utf-8')
for talk_line in f_talk:  
    talk_words = talk_line.replace('\n', '')
    split_talk = talk_words.split(':')
    talk_dic[split_talk[0]]=split_talk[1]

#compile doc-word set
f = open('./new_SymptomToOutpatient.txt', encoding = 'utf-8').read()
splitF = f.split(':')
words = [w.replace('\n', '') for w in splitF]
depgroup =set()
for dep in words:
    depname = dep.split(',')
    depgroup.update(depname[:-1])
#print(depgroup)

new_f = open('new_SymptomToOutpatient.txt', encoding = 'utf-8')
doc_list= []
for deps in depgroup: 
    slist=[]
    new_f.seek(0)
    for line in new_f:
        sym_dep = line.split(":")
        depset = sym_dep[1].split(",")
        for dep in depset:
            if(dep==deps):
                slist.append(sym_dep[0])
    joined = " ".join(slist)
    doc_list.append(joined)

wf = open('word_doc.txt','w', encoding = 'utf-8')
for dw in doc_list:
    wf.write(dw + "\n")

wf_new = open('word_doc.txt', encoding = 'utf-8')  
doccontent=[]
for line in wf_new:
    line = line.replace("\n","")
    doccontent.append(line)
    
vectorizer = TfidfVectorizer()                  
tfidf = vectorizer.fit_transform(doccontent).toarray()            
vocab = vectorizer.get_feature_names()

#age question
age_answer = input('How old are you? ')
if int(age_answer) < 18:
    print("衛生署宣導：十八歲以下的人士，仍建議看兒科，台灣兒科醫學會也表示，18 歲前的兒童與青少年，身心皆處於發育階段，發病症狀與治療方式與成人有所差異，請選擇至兒科院所就診")
        
sex_answer = input('are you male or female (male= 1 or female=2) ')
if sex_answer == "1":
    sex = "Male"
    print("先生您好!")
if sex_answer == "2":
    sex = "Female" 
    print("小姐您好!")
    
test_set=[]
print("症狀:" + str(vocab))
initial_sym = input('please write the initial symptons (splited by space):')
print("您的初始症狀是:" + initial_sym)
test_set.append(initial_sym)
query_set = []
split_sym=test_set[0].split(" ")
for symid in range(len(split_sym)):
    query_set.append(split_sym[symid])

if sex=="Female":
    query_set.append("睾丸疼痛")
    query_set.append("遺精")
else:
    query_set.append("乳房腫塊")
    query_set.append("白帶")
    query_set.append("閉經")
    query_set.append("痛經")
    query_set.append("宮頸糜爛")
    query_set.append("流產")

testVectorizerArray = vectorizer.transform(test_set).toarray()
cossim = cosine_similarity(testVectorizerArray, tfidf)
#print(cossim)

nonzero_list = np.nonzero(cossim)
argsort = np.argsort(cossim)

a = [argsort[0][i] for i in range(len(argsort[0])) if argsort[0][i] in nonzero_list[1]]
for i in a:
    print(list(depgroup)[i])

newlist = []
for depid in a:
    newlist.append(doccontent[depid])

tfvectorizer = TfidfVectorizer()                       
X = tfvectorizer.fit_transform(newlist)
features = tfvectorizer.get_feature_names()

def top_tfidf_feats(row, features, top_n=10):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [features[i] for i in topn_ids]
    df = [(features[i], row[i]) for i in topn_ids]
    feature_df = pd.DataFrame(df)
    feature_df.columns = ['feature', 'tfidf']
    print(feature_df)
    return top_feats

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.max(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

top_feats = top_mean_feats(X,features, top_n=20)
choice = 0
answer = "-1"
if len(a)==1: 
    choice=6
    print("recommand:" + str(list(depgroup)[a[len(a)-1]]) )
while(choice<6):
    choice +=1
    print("query_set" + str(query_set))
    if answer=="1":
        print("answer 1" + test_set[0])
        test_set[0] = test_set[0] + " " + query
        print("test_set" + str(test_set))
        # rerun newlist
        testVectorizerArray = vectorizer.transform(test_set).toarray()
        cossim = cosine_similarity(testVectorizerArray, tfidf)
        #print(cossim)
        nonzero_list = np.nonzero(cossim)
        argsort = np.argsort(cossim)
        a = [argsort[0][i] for i in range(len(argsort[0])) if argsort[0][i] in nonzero_list[1]]
        for i in a:
            print(list(depgroup)[i]) 
    
        newlist = []    
        for depid in a:
            newlist.append(doccontent[depid])
        #print(newlist)
        tfvectorizer = TfidfVectorizer()                       
        X = tfvectorizer.fit_transform(newlist)
        features = tfvectorizer.get_feature_names()
        #print('Transform Vectorizer to test set' , str(X.shape))
        top_feats = top_mean_feats(X,features, top_n=10)
        query = ""
        for i in range(len(top_feats)):
            if(top_feats[i] not in query_set):
                query = top_feats[i]
                query_talk = talk_dic[query]
                query_set.append(top_feats[i])
                break
     
    # check top deps > definded_count --> choice to false
    elif answer=="-1":
        print("answer -1")
        query = ""
        # check top deps > definded_count --> choice to false
        for i in range(len(top_feats)):
            if(top_feats[i] not in query_set):
                query = top_feats[i]
                query_talk = talk_dic[query]
                query_set.append(top_feats[i])
                break
    else:
        print("please type (1 or -1)")
        choice -=1
    answer = input(query_talk+'(1 or -1) ')
    
if int(age_answer)>18 and str(list(depgroup)[a[len(a)-1]])=="小兒科":
    print("recommand:家醫科" )
else:
    print("recommand:" + str(list(depgroup)[a[len(a)-1]]) ) 


# In[ ]:



