import numpy as np

class SmiTok(object):
    def __init__(self):
        atoms = [
                'H', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na','Mg', 
                'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 
                'Br', 'I'
                ]
        special = [
                '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
                '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
                'p', 'se', 'te', 'c', 'n', 'o', 's', '[nH]','\n','l','r','[S+]',
                '[O-]','[N+]','[N-]'
                ]
        self.table = sorted(atoms+special, key=len, reverse=True)
        self.table_len = len(self.table)

    def tokenize(self,smiles):
        N = len(smiles)
        i = 0
        token = []
        while i<N:
            for j in range(self.table_len):
                symbol = self.table[j]
                if symbol == smiles[i:i+len(symbol)]:
                    token.append(symbol)
                    i += len(symbol)
                    break
        return token

    def findsamechar(self,smifile):
        f = open(smifile,'r')
        los = f.readlines()
        setofchar = set()
        for i in los:
            a = set(list(i))
            setofchar = setofchar|s
        return setofchar

def findsametok(trainfile):
    ST = SmiTok()
    f = open(trainfile,'r')
    l = f.readlines()
    f.close()
    a = set()
    for smi in l:
        a = a|set(ST.tokenize(smi))
    return list(a)

def build_dic(list_of_tok):
    size = len(list_of_tok)
    list_of_tok.sort()
    dic_of_tok = dict(zip(list_of_tok, range(size)))
    return dic_of_tok

def lookup_dic(list_of_tok):
    size = len(list_of_tok)
    list_of_tok.sort()
    dic_of_id = dict(zip(range(size), list_of_tok))
    return dic_of_id

def tok_to_id(tokens, dic_of_tok):
    list_of_id  = [dic_of_tok[token] for token in tokens]
    return list_of_id

if __name__ == "__main__":
    import pickle
    trainfile = 'train_tl.txt'
    f = open('tokdict.pickle', 'r')
    dicoftokens = pickle.load(f)
    f.close()
    
    f = open(trainfile,'r')
    smi_list = f.readlines()
    f.close()
    maxlength = 0
    minlength = 1000
    tralist = []
    ST = SmiTok()
    for i in smi_list1:
        tok_list = ST.tokenize(i)
        print i
        s = len(tok_list)
        if s>maxlength:maxlength=s
        if s<minlength:minlength=s
        for j in tok_list:
            tralist.append(j)
    trainid = tok_to_id(tralist, dicoftokens)
    
    f = open('train_tl.pickle', 'wb')
    pickle.dump(trainid, f)
    f.close()
    print 'max: %s tokens \t min: %s tokens'%(maxlength,minlength)
    print 'size of ids: %s'%(len(trainid))
