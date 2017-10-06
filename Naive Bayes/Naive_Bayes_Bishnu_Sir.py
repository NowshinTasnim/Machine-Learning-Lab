'''Author: Bishnu Sarker, dept. of Computer Sciecne and Engineering, Khulna  -
University of Engineering & Technology'''
import random
import math

def write_object(object,filename):
    '''
    To write a python list object in external file
    :param object: Object prefereably a List object to be written in a  -
    file
    :param filename: Filename in which data will be written
    :return: NA
    '''
    with open(filename,'w') as W:
            for item in object:
                W.write(str(item)+'\n')

def read_review(filename):
    '''
    Reading raw review and prepraring dataset seperating pos and neg data
    :param filename: Dataset file name
    :return:tuple, positive dataset and negative dataset
    '''
    pos_dataset=[]
    neg_dataset=[]
    with open(filename) as R:
        for line in R:
            #print(line)
            line=line.split()
            if line[0]=="Neg":
                neg_dataset.append(list(line[1:]))
            else:
                pos_dataset.append(list(line[1:]))
    write_object(pos_dataset,"positve.txt")
    write_object(neg_dataset,"negative.txt")
    return (pos_dataset,neg_dataset)

def split_test_train(pos, neg):
    '''
    Splits the datasets into train and test set
    :param pos: positive dataset
    :param neg: negative dataset
    :return: postive training, negative training, positive testing and  -
    negative training
    '''
    pl=len(pos)
    nl=len(neg)

    pSample_train=random.sample(pos,int(pl*0.80))
    nSample_train=random.sample(neg,int(nl*0.80))
    pSample_test=random.sample(pos,int(pl*0.20))
    nSample_test=random.sample(neg,int(nl*0.20))

    return (pSample_train,nSample_train,pSample_test,nSample_test)

def build_Vocab(pTrain,nTrain):
    '''
    Building a vocabulary V from taining dataset
    :param pTrain: positive training dataset
    :param nTrain: negative training dataset
    :return: Vocabulary of the form {"Token":(p, n)} p for positve  -
    frequency and n for negative frequency
    '''
    Vocabulary={}
    for sample in pTrain:
        for word in sample:
            word=word.lower().strip("’!.:)(?-")
            if word == "":
                continue
            if Vocabulary.__contains__(word):
                freq,T=Vocabulary[word]
                freq=freq+1
                Vocabulary[word]=(freq,T)
            else:
                Vocabulary.__setitem__(word,(1,0))
    for sample in nTrain:
        for word in sample:
            word = word.lower().strip("’!.:)(?-[]+")
            if word=="":
                continue
            if Vocabulary.__contains__(word):
                T,freq=Vocabulary[word]
                freq=freq+1
                Vocabulary[word]=(T,freq)
            else:
                Vocabulary.__setitem__(word,(0,1))
    print_dic(Vocabulary)
    return Vocabulary

def Maxmimum_likelihood_Estimation(pTrain,nTrain):
    '''
    Compute the required probabilities using maximum likelihood estimation.
    :param pTrain:
    :param nTrain:
    :return:
    '''
    pN=len(pTrain)
    nN=len(nTrain)
    N=pN+nN
    pCount=0
    nCount=0
    vocab=build_Vocab(pTrain,nTrain)
    V=len(vocab)
    for words in vocab:
        pCount=pCount+vocab[words][0]
        nCount=nCount+vocab[words][1]
    return {"vocab":vocab,"pN":pN,"nN":nN,"N":N,"V":V,"pC":pCount,"nC": nCount}


def NB_classifier(test_sample,Vocab,pN,nN,N,V,pCount,nCount):
    '''
    Naive bayes classifier
    :param test_sample:
    :param Vocab:
    :param pN:
    :param nN:
    :param N:
    :param V:
    :param pCount:
    :param nCount:
    :return:
    '''
    #tokens=test_sample.strip().split()
    N=float(N)
    prob_pos=pN/N
    prob_neg=nN/N
    pos_prob=0.0
    neg_prob=0.0
    for token in test_sample:
        token=token.lower().strip("’!.:)(?-[]+")
        #computing p(token|pos)
        if token=="":
            continue
        if Vocab.__contains__(token):
            pos_prob=pos_prob+math.log10(Vocab[token][0]+1.0/(pCount+V))
            neg_prob=neg_prob+math.log10(Vocab[token][1]+1.0/(nCount+V))
        else:
            pos_prob = pos_prob + math.log10( 1.0 / (pCount + V))
            neg_prob = neg_prob + math.log10( 1.0 / (nCount + V))

    pP=math.log10(prob_pos)+pos_prob
    nP=math.log10(prob_neg)+neg_prob
    return (pP,nP)

def testing(pTest,nTest,Param):
    '''
    Find the probabale class for a test sample
    :param pTest:
    :param nTest:
    :param Param:
    :return:
    '''
    with open("Rsult.txt",'w') as W:
        for item in pTest:
            pP,nP=NB_classifier(item,Param["vocab"],Param["pN"],Param["nN"],Param["N"],Param["V"],Param["pC"],Param["nC"])
            print("Prediction-->\tpos:"+str(pP)+" \tNeg:"+str(nP)+"\tActual-->Positive")
            W.write("Prediction--> pos:"+str(pP)+" \tNeg:"+str(nP)+"\tActual-->Positive\t"+str(item)+'\n')
        for item in nTest:
            pP, nP = NB_classifier(item,Param["vocab"],Param["pN"],Param["nN"],Param["N"],Param["V"],Param["pC"],Param["nC"])
            print("Prediction--> pos:" + str(pP) + " Neg:" + str(nP)+"Actual-->Negative")
            W.write("Prediction--> pos:" + str(pP) + " \tNeg:" + str(nP) + " \tActual-->Negative\t"+str(item)+"\n")
def print_dic(D):
    with open("Vocab.txt",'w') as W:
        for d in D.keys():
            W.write(d+"-->"+" Pos:"+str(D[d][0])+" Neg:"+str(D[d][1])+'\n')

if __name__ == '__main__':
    pos,neg=read_review("AppleReview.txt")
    pos_train,neg_train,pos_test,neg_test=split_test_train(pos,neg)
    Params=Maxmimum_likelihood_Estimation(pos_train,neg_train)
    print (Params)
    testing(pos_test,neg_test,Params)