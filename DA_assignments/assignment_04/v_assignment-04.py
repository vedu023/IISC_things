import numpy as np
import time
import scipy as sp


"""Starting and ending locations (indices) of red and green exons in the reference sequence - Begins

1. Red Exon Locations
"""
RedExonPos = np.array([
    [149249757, 149249868], # R1
    [149256127, 149256423], # R2
    [149258412, 149258580], # R3
    [149260048, 149260213], # R4
    [149261768, 149262007], # R5
    [149264290, 149264400]  # R6
    ])
"""
2. Green Exon Locations
"""
GreenExonPos = np.array([
    [149288166, 149288277], # G1
    [149293258, 149293554], # G2
    [149295542, 149295710], # G3
    [149297178, 149297343], # G4
    [149298898, 149299137], # G5
    [149301420, 149301530]  # G6
    ])
"""
Starting and ending locations (indices) of red and green exons in the reference sequence - Ends
"""   
redleft = RedExonPos[:,0]
redright = RedExonPos[:,1]
greenleft = GreenExonPos[:,0]
greenright = GreenExonPos[:,1] 

def loadLastCol(filename):  
    lastCol = ''.join(np.loadtxt(filename, dtype=str))
    return lastCol  

def loadRefSeq(filename):
    ref = ''.join(np.loadtxt(filename, dtype=str)[1:])
    return ref  

def loadReads(filename):
    reads = np.loadtxt(filename, dtype=str)
    return reads 

def loadMapToRefSeq(filename):
    Map = np.loadtxt(filename, dtype=int)
    return Map  


class Rank:
    def __init__(self,bwt,k = 200):
        self.k = k
        self.l = len(bwt)
        rem = k - (self.l % k)
        self.bwt = bwt
        n_bwt = np.array(list(bwt + 'Z'*rem)).reshape(-1,k)

        unique, counts = np.unique(n_bwt, return_counts = True)
        unique = unique[1:]
        counts = counts[1:]
        if rem:
            unique = unique[:-1]
            counts = counts[:-1]
        
        counts = counts.cumsum()
        self.s_idx = {'A' : 0, 'C' : counts[0], 'G' : counts[1], 'T' : counts[2]}
        self.e_idx = {'A' : counts[0] - 1, 'C' : counts[1] - 1, 'G' : counts[2] - 1, 'T' : counts[3] - 1}
        
        self.block_cum = {}    
        for c in unique:
            temp = (n_bwt == c).sum(axis=1).cumsum()
            self.block_cum[c] = temp

        del(n_bwt, counts)

    def lcr(self, c, idx):
        assert(idx < self.l)
        t1 = (idx + 1) // self.k
        rank = self.block_cum[c][t1 - 1] if t1-1 >= 0 else 0
        j = t1 * self.k
        
        while j <= idx:  
            rank += (self.bwt[j] == c)
            j += 1
        return rank

    def fci(self, c, rank):
        assert(rank > 0)
        return self.s_idx[c] + rank - 1

    def fcfi(self,c):
        return self.s_idx[c]

    def fcli(self,c):
        return self.e_idx[c]


def exact_match(a,shred):
    match, j = 0, 0
    l = len(shred)
    while j < l:
        match += (ref[a] != shred[j])
        a += 1
        j += 1
    return match 


def mis_match(i,shred, max_mm = 2):
    mm, j = 0, 0
    l = len(shred)
    while j < l and mm <= max_mm:
        mm += (ref[i] != shred[j])
        i += 1
        j += 1
    return mm <= max_mm


def batch_cal(shred):
    l = len(shred)
    s,e = fcfi(shred[-1]),fcli(shred[-1])
    i = l - 2
    while i > -1:
        c = shred[i]
        if s == e:
            break
        srank, erank = lcr(c, s), lcr(c, e)
        srank += (1 if srank < erank and LastCol[s] != c else 0)
        s, e = fci(c, srank), fci(c, erank)
        i -= 1
    
    return s, e, i + 1


def MatchReadToLoc(read):
    """
    Input: a read (shred)
    Output: list of potential locations at which the read may match the reference sequence. 
    Refer to example in the README file.
    IMPORTANT: This function must also handle the following:
        1. cases where the read does not match with the reference sequence with less than 2 error
        2. any other special case that you may encounter
    """
    # function body - Begins

    read = read.replace('N','A')
    r_read = read[::-1]
    r_read = r_read.translate(str.maketrans('ACGT','TGCA'))
    s1, e1, p1 = batch_cal(read)
    s2, e2, p2 = batch_cal(r_read)

    positions = []
    
    read = read[:p1]
    r_read = r_read[:p2]
    for i in range(s1, e1 + 1):
        idx = Map[i] - p1
        if idx  > -1 and mis_match(idx, read):
            positions.append(idx)

    for i in range(s2, e2 + 1):
        idx = Map[i] - p2
        if idx  > -1 and mis_match(idx, r_read):
            positions.append(idx)

    return positions # list of potential locations at which the read may match the reference sequence.


def WhichExon(positions):
    """
    Input: list of potential locations at which the read may match the reference sequence.
    Output: Update(increment) to the counts of the 12 exons
    IMPORTANT: This function must also handle the following:
        1. cases where the read does not match with the reference sequence
        2. cases where there are more than one matches (match at two exons)
        3. any other special case that you may encounter
    """
    R,G = np.zeros(6), np.zeros(6)
    
    # function body - Begins
    for each in positions:
        t1 = ((redleft <= each) & (each <= redright))
        t2 = ((greenleft <= each) & (each <= greenright))
        
        if np.sum(t1) == 1 and np.sum(t2) == 1:
            R[t1] += 0.5
            G[t2] += 0.5
        else:
            R[t1] += 1
            G[t2] += 1

    # function body - Ends

    return np.concatenate([R,G])


def ComputeProb(ExonMatchCounts):
    """
    Input: The counts for each exon
    Output: Probabilities of each of the four configurations (a list of four real numbers)
    """
    # function body - Begins
    
    a = ExonMatchCounts
    r = a[1:5]
    g = a[7:11]
    n = r+g

    def prob(p,k):
        prob = sp.special.comb(n[k], r[k])*(p^r[k])*((1-p)^g[k])
        return prob
    
    p1 = [1/3,1/3,1/3,1/3]
    p2 = [1/2,1/2,0,0]
    p3 = [1/4,1/4,1/2,1/2]
    p4 = [1/4,1/4,1/4,1/2]

    P1 = [np.log(prob(p1[i], i)) for i in range(4)].sum()
    P2 = [np.log(prob(p2[i], i)) for i in range(4)].sum()
    P3 = [np.log(prob(p3[i], i)) for i in range(4)].sum()
    P4 = [np.log(prob(p4[i], i)) for i in range(4)].sum()

    # function body - ends
    return [P1, P2, P3, P4]

def BestMatch(ListProb):
    print(f'prob of configs...P1 = {ListProb[0]}\nP2 = {ListProb[1]}\nP3 = {ListProb[2]}\nP4 = {ListProb[3]}')
    return np.argmax(ListProb) + 1



if __name__ == "__main__":
    # load all the data files
    # prefix = '../data/'
    print('loading...')
    t1 = time.time()
    prefix = 'chrX_bwt/'
    LastCol = loadLastCol(prefix + "chrX_last_col.txt")  
    ref = loadRefSeq(prefix + "chrX.fa")  
    reads = loadReads(prefix + "reads")  
    Map = loadMapToRefSeq(prefix + "chrX_map.txt")  
    print(" files loaded...")
    t2 = time.time()
    print(f"fileloaing time is {(t2 - t1)/60 :0.3f} min")

    rank = Rank(LastCol, k = 5)    
    lcr = rank.lcr
    fci = rank.fci
    fcfi = rank.fcfi
    fcli = rank.fcli

    print("start computing...")
    ExonMatchCounts = np.zeros(12) 
    l = len(reads)

    t1 = time.time()
    for i in range(l):  
        read = reads[i]
        positions = MatchReadToLoc(read) 
        ExonMatchCounts += WhichExon(positions)  
    
    t2 = time.time()
    print(f"Runtime of the program is {(t2 - t1)/60 :0.3f} min")
        
    print(ExonMatchCounts)
    ListProb = ComputeProb(ExonMatchCounts)  
    MostLikely = BestMatch(ListProb)  
    print("Configuration %d is the best match"%MostLikely)



