import FSM
import util

def readData(filename):
    h = open(filename, 'r')
    words = []
    segmentations = []
    for l in h.readlines():
        a = l.split()
        if len(a) == 1:
            words.append(a[0])
            segmentations.append(None)
        elif len(a) == 2:
            words.append(a[0])
            segmentations.append(a[1])
    return (words, segmentations)

def evaluate(truth, hypothesis):
    I = 0
    T = 0
    H = 0
    for n in range(len(truth)):
        if truth[n] is None: continue 
        t = truth[n].split('+')
        allT = {}
        cumSum = 0
        for ti in t:
            cumSum = cumSum + len(ti)
            allT[cumSum] = 1

        h = hypothesis[n].split('+')
        allH = {}
        cumSum = 0
        for hi in h:
            cumSum = cumSum + len(hi)
            allH[cumSum] = 1

        T = T + len(allT) - 1
        H = H + len(allH) - 1
        for i in allT.iterkeys():
            if allH.has_key(i):
                I = I + 1
        I = I - 1
        
    Pre = 1.0
    Rec = 0.0
    Fsc = 0.0
    if I > 0:
        Pre = float(I) / H
        Rec = float(I) / T
        Fsc = 2 * Pre * Rec / (Pre + Rec)
    return (Pre, Rec, Fsc)

def stupidChannelModel(words, segmentations):
    # figure out the character vocabulary
    vocab = util.Counter()
    for w in words:
        for c in w:
            vocab[c] = vocab[c] + 1

    # build the FST    
    fst = FSM.FSM(isTransducer=True, isProbabilistic=True)
    fst.setInitialState('s')
    fst.setFinalState('s')
    for w in words:
        for c in w:
            fst.addEdge('s', 's', c, c, prob=1.0)    # copy the character
    fst.addEdge('s', 's', '+', None, prob=0.1)       # add a random '+'
    return fst

def stupidSourceModel(segmentations):
    # figure out the character vocabulary
    vocab = util.Counter()
    for s in segmentations:
        for c in s:
            vocab[c] = vocab[c] + 1
    # convert to probabilities
    vocab.normalize()

    # build the FSA
    fsa = FSM.FSM(isProbabilistic=True)
    fsa.setInitialState('s')
    fsa.setFinalState('s')
    for c,v in vocab.iteritems():
        fsa.addEdge('s', 's', c, prob=v)
    return fsa

def bigramSourceModel(segmentations): return bigramSourceModel2(segmentations)[0]

# This version of bigramSourceModel returns the vocab and lm objects, not just the fsm
def bigramSourceModel2(segmentations):
    # compute all bigrams
    lm = {}
    vocab = {}
    vocab['end'] = 1
    for s in segmentations:
        prev = 'start'
        for c in s:
            if not lm.has_key(prev): lm[prev] = util.Counter()
            lm[prev][c] = lm[prev][c] + 1
            prev = c
            vocab[c] = 1
        if not lm.has_key(prev): lm[prev] = util.Counter()
        lm[prev]['end'] = lm[prev]['end'] + 1

    # smooth and normalize
    for prev in lm.iterkeys():
        for c in vocab.iterkeys():
            lm[prev][c] = lm[prev][c] + 0.2   # add 0.5 smoothing
        lm[prev].normalize()

    # convert to a FSA
    fsa = FSM.FSM(isProbabilistic=True)
    fsa.setInitialState('start')
    fsa.setFinalState('end')
    for h in lm:
        for c in lm[h]:
            if c == 'end':
                fsa.addEdge(h, c, None, prob=lm[h][c]) # need esp
            else:
                fsa.addEdge(h, c, c, prob=lm[h][c])

    return (fsa,vocab,lm)

def buildSegmentChannelModel(words, segmentations):
    segments = set()
    charDict = {}
    fst = FSM.FSM(isTransducer=True, isProbabilistic=True)
    fst.setInitialState('start')
    fst.setFinalState('end')
    fst.addEdge('endseg', 'end', None, None)
    fst.addEdge('endseg', 'start', '+', None)
    #add chunks to our segment set
    for s in segmentations:
        chunks = s.split('+')
        for c in chunks:
            segments.add(c)
    
    for s in segments:
        fst.addEdgeSequence('start', 'endseg', s)
        
    #iterate over all characters we've seen in bengali and add some small prob to 'smooth' unseen segments  
    for word in words:
        for char in word:
            if not char in charDict:
                #print char
                charDict[char] = 1
                fst.addEdge('start', 'start', char, char, prob=0.1)

    fst.addEdge('start', 'start', '+', None, prob=0.1)
    fst.addEdge('start', 'end', None, None, prob=0.1)
    
    return fst


def fancySourceModel(segmentations): return fancySourceModel2(segmentations)[0]
    
def fancySourceModel2(segmentations):
    #lm = [[[0 for x in range(1000)] for x in range(1000)] for x in range(1000)]
    lm = {}
    bi = {}
    vocab = {}
    vocab['end'] = 1
    for s in segmentations:
        prev = 'start'
        prev1 = 'start'
        for c in s:
            
            if not prev1 in lm:
                lm[prev1] = {}
            if not prev in lm[prev1]:
                #lm[prev1][prev] = util.Counter()
                lm[prev1][prev] = {}
            if not c in lm[prev1][prev]:
                lm[prev1][prev][c] = 0
            lm[prev1][prev][c] = lm[prev1][prev][c] + 1
            if not prev1 in bi:
                bi[prev1] = {}
            if not prev in bi[prev1]:
                bi[prev1][prev] = 0
            bi[prev1][prev] = bi[prev1][prev] +1
            prev1 = prev
            prev = c
            vocab[c] = 1
                #if not bi.has_key(prev): bi[prev] = util.Counter()
        if not prev1 in lm:
                lm[prev1] = {}
        if not prev in lm[prev1]:
                lm[prev1][prev] = {}
        if not 'end' in lm[prev1][prev]:
            lm[prev1][prev]['end'] = 0
        if not prev1 in bi:
            bi[prev1] = {}
        if not prev in bi[prev1]:
            bi[prev1][prev] = 0
        lm[prev1][prev]['end'] = lm[prev1][prev]['end'] + 1
        bi[prev1][prev] = bi[prev1][prev] + 1
    # smooth and normalize
    for prev1 in lm:
        for prev in lm[prev1]:
            for c in lm[prev1][prev]:
                lm[prev1][prev][c] = (lm[prev1][prev][c] + 0.1)/bi[prev1][prev]
    
    # convert to a FSA
    fsa = FSM.FSM(isProbabilistic=True)
    fsa.setInitialState('start')
    fsa.setFinalState('end')

    for p in lm:
        for h in lm[p]:
            fsa.addEdge('start',p+h, p+h)
            for c in lm[p][h]:
                if c == 'end':
                    fsa.addEdge(p+h, 'end', None, prob=lm[p][h][c]) # need esp
                else:
                    fsa.addEdge(p+h, h+c, c, prob=lm[p][h][c])
    
    for p in lm:
        for h in lm[p]:
            for c in lm[p][h]:
                print lm[p][h][c]
    return (fsa, lm)

def fancyChannelModel(words, segmentations):
    raise Exception("fancyChannelModel not defined")

    
def runTest(trainFile='bengali.train', devFile='bengali.test', channel=stupidChannelModel, source=stupidSourceModel):
    (words, segs) = readData(trainFile)
    (wordsDev, segsDev) = readData(devFile)
    fst = channel(words, segs)
    fsa = source(segs)

    preTrainOutput = FSM.runFST([fsa, fst], wordsDev, quiet=True)
    for i in range(len(preTrainOutput)):
        if len(preTrainOutput[i]) == 0: preTrainOutput[i] = words[i]
        else:                           preTrainOutput[i] = preTrainOutput[i][0]
    preTrainEval   = evaluate(segsDev, preTrainOutput)
    print 'before training, P/R/F = ', str(preTrainEval)

    fst.trainFST(words, segs)

    postTrainOutput = FSM.runFST([fsa, fst], wordsDev, quiet=True)
    for i in range(len(postTrainOutput)):
        if len(postTrainOutput[i]) == 0: postTrainOutput[i] = words[i]
        else:                            postTrainOutput[i] = postTrainOutput[i][0]
    postTrainEval   = evaluate(segsDev, postTrainOutput)
    print 'after  training, P/R/F = ', str(postTrainEval)
    
    return postTrainOutput

def saveOutput(filename, output):
    h = open(filename, 'w')
    for o in output:
        h.write(o)
        h.write('\n')
    h.close()
    

if __name__ == '__main__':

    print '/*******************************/'
    print '/*******seg channel only********/'
    print '/*******************************/'
    #runTest(channel=buildSegmentChannelModel)

    print '/*******************************/'
    print '/*******bigram and segment******/'
    print '/*******************************/'
    #runTest(source=bigramSourceModel,channel=buildSegmentChannelModel)

    print '/*******************************/'
    print '/*******bigram only*************/'
    print '/*******************************/'
    #runTest(source=bigramSourceModel)
    runTest(source = fancySourceModel, channel = buildSegmentChannelModel)

    
