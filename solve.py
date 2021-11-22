import re
import sys

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm

class Vigenere:
    def cipherText(self, string, key):
        cipher_text = []
        i = 0
        for s in string:
            if i >= len(key): i = 0
            x = (ord(s)+ord(key[i])-2*ord('a'))%26
            x += ord('a')
            cipher_text.append(chr(x))
            i += 1
        return(''.join(cipher_text))

    def decryptText(self, string, key):
        orig_text = []
        i = 0
        for s in string:
            if i >= len(key): i = 0
            x = (ord(s)-ord(key[i])+26)%26
            x += ord('a')
            orig_text.append(chr(x))
            i += 1
        return(''.join(orig_text))

    def cipherNum(self, clear, key):
        return (clear+self.genKey(key, clear.shape))%26

    def decryptNum(self, cipher, key):
        return (cipher-self.genKey(key, cipher.shape)+26)%26

    def genKey(self, key, shape):
        cN = shape[0] if len(shape)==1 else shape[1]
        return np.tile(key, (cN//len(key))+1)[:cN]

def alp2num(s):
    assert s==s.lower()
    assert s==s.replace(' ', '')
    c2n = lambda x: ord(x)-ord('a')
    return np.array([c2n(ss) for ss in s])

def num2alp(s):
    if not (isinstance(s,np.ndarray) or isinstance(s,list)):
        s = [s]
    n2c = lambda x: chr(x+ord('a'))
    return ''.join(map(n2c, s))

def logScaled(df):
    df = df.T
    df['log'] = df.apply(lambda x: np.log(x))
    bmax,bmin = df.log.max(), df.log.min()

    # skaler log verdiene til å være mellom 1 og 1000
    scale = lambda x: int((x-bmin)/(bmax-bmin)*99+1)
    df['log_scaled'] = df.log.apply(scale)
    return df.log_scaled.to_dict()

def chunk(targ, size, cont=False):
    if cont:
        targ = np.array([targ[i:i+size] for i in range(0, len(targ)-size+1)])
    else:
        mod = len(targ)%size
        if mod != 0: targ=targ[:-mod]
        targ = targ.reshape(-1, size) 
    return targ

def cleanText(text):
    regex = re.compile('[^a-zA-Z]')
    text = regex.sub('', text).lower()
    return text

def biChunk(chunks):
    n = chunks.shape[1]
    chunks = np.hstack((chunks,chunks[:,:1]))
    biChunks = []
    for i in range(n):
        biChunks.append(chunks[:,i:i+2])
    return biChunks

V = Vigenere()

if len(sys.argv) == 1:
    print('Usage: python3 solve.py <cipher>')
    print('or: python3 solve.py <text> <key>')
    exit()
elif len(sys.argv) == 2:
    cipher = alp2num(cleanText(sys.argv[1]))
elif len(sys.argv) == 3:
    print('original text:')
    print(sys.argv[1])
    cipher = V.cipherNum(alp2num(cleanText(sys.argv[1])), alp2num(cleanText(sys.argv[2])))

print('cipher:')
print(num2alp(cipher))

bigrams = logScaled(pd.read_feather('bigrams.feather'))

allBigrams = bigrams.keys()
tmpResults = []
for kl in tqdm(range(3,30)):
    chunks = chunk(cipher, kl)
    tmplist = []
    for cnk in biChunk(chunks):
        best = 0
        for bigram in allBigrams:
            decrypted = V.decryptNum(cnk, alp2num(bigram))
            fitness = 0
            for b in (num2alp(e1)+num2alp(e2) for e1,e2 in decrypted):
                try: q = bigrams[b]
                except: q = 0
                fitness += q
            if fitness > best:
                best = fitness
                bigramHolder = bigram
        tmplist.append([bigramHolder,best])
    tmpResults.append(tmplist)

results = []
for c,r in enumerate(tmpResults,3):
    candkey = r[0][0][0] if r[0][1] > r[-1][1] else r[-1][0][1]
    for i in range(len(r)-1):
        t1,t2 = r[i],r[i+1]
        candkey += t1[0][1] if t1[1] > t2[1] else t2[0][0]
    results.append(candkey)

best = 0
bestKeys = {}
for k in results:
    fitness = 0
    dc = num2alp(V.decryptNum(cipher, alp2num(k)))
    for i in range(len(cipher)-1):
        try: q = bigrams[dc[i:i+2]]
        except: q = 0
        fitness += q
    bestKeys[k] = fitness-(len(k)*100)

print("10 beste keys:")
print(sorted(bestKeys.items(), reverse=True, key=lambda x: x[1])[:10])
