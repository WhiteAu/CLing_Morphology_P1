import FSM
import util

vocabulary = ['panic', 'picnic', 'ace', 'pack', 'pace', 'traffic', 'lilac', 'ice', 'spruce', 'frolic']
suffixes   = ['', '+ed', '+ing', '+s']

def buildSourceModel(vocabulary, suffixes):
    # we want a language model that accepts anything of the form
    # *   w
    # *   w+s
    fsa = FSM.FSM()
    fsa.setInitialState('start')
    fsa.setFinalState('end')
    
    for word in vocabulary:
        fsa.addEdgeSequence('start', '1', word)

    fsa.addEdge('1', 'end', None)           # if there is no suffix

    for suffix in suffixes:
        if len(suffix) > 0:                 # skip if no suffix
            fsa.addEdgeSequence('1', 'end', suffix)


    return fsa

def buildChannelModel():
    # this should have exactly the same rules as englishMorph.py!
    fst = FSM.FSM(isTransducer=True)
    fst.setInitialState('start')
    fst.setFinalState('end')

    # we can always get from start to end by consuming non-+
    # characters... to implement this, we transition to a safe state,
    # then consume a bunch of stuff
    fst.addEdge('start', 'safe', '.', '.')
    fst.addEdge('safe',  'safe', '.', '.')
    fst.addEdge('safe',  'safe2', '+', None)
    fst.addEdge('safe2', 'safe2', '.', '.')
    fst.addEdge('safe',  'end',  None, None)
    fst.addEdge('safe2',  'end',  None, None)
    
    # implementation of rule 1
    fst.addEdge('start' , 'rule1' , None, None)   # epsilon transition
    fst.addEdge('rule1' , 'rule1' , '.',  '.')    # accept any character and copy it
    fst.addEdge('rule1' , 'rule1b', 'e',  None)   # remove the e
    fst.addEdge('rule1b', 'rule1c', '+',  None)   # remove the +
    fst.addEdge('rule1c', 'rule1d', 'e',  'e')    # copy an e ...
    fst.addEdge('rule1c', 'rule1d', 'i',  'i')    #  ... or an i
    fst.addEdge('rule1d', 'rule1d', '.',  '.')    # and then copy the remainder
    fst.addEdge('rule1d', 'end' , None, None)   # we're done

    # implementation of rule 2
    fst.addEdge('start' , 'rule2' , '.', '.')     # we need to actually consume something
    fst.addEdge('rule2' , 'rule2' , '.', '.')     # accept any character and copy it
    fst.addEdge('rule2' , 'rule2b', 'e', 'e')     # keep the e
    fst.addEdge('rule2b', 'rule2c', '+', None)    # remove the +
    for i in range(ord('a'), ord('z')):
        c = chr(i)
        if c == 'e' or c == 'i':
            continue
        fst.addEdge('rule2c', 'rule2d', c, c)     # keep anything except e or i
    fst.addEdge('rule2d', 'rule2d', '.', '.')     # keep the rest
    fst.addEdge('rule2d', 'end' , None, None)   # we're done

    # implementation of rule 3
    fst.addEdge('start' , 'rule3' , '.', '.')     # not sure if eps or not
    fst.addEdge('rule3' , 'rule3' , '.', '.')     
    for c in ['a','e','i','o','u']:
        fst.addEdge('rule3', 'rule3b', c, c)     
    fst.addEdge('rule3b', 'rule3c', 'c', 'c')     
    fst.addEdge('rule3c', 'rule3d', '+', 'k')     
    fst.addEdge('rule3d', 'rule3e', 'e', 'e')     
    fst.addEdge('rule3d', 'rule3e', 'i', 'i')     
    fst.addEdge('rule3e', 'rule3e', '.', '.')     
    fst.addEdge('rule3e', 'end' , None, None)   


    return fst

def simpleTest():
    fsa = buildSourceModel(vocabulary, suffixes)
    fst = buildChannelModel()

    print "==== Trying source model on strings 'ace+ed' ===="
    #was [fsa]
    output = FSM.runFST([fsa], ["ace+ed"])
    print "==== Result: ", str(output), " ===="

    print "==== Trying source model on strings 'panic+ing' ===="
    #was [fsa]
    output = FSM.runFST([fsa], ["panic+ing"])
    print "==== Result: ", str(output), " ===="
    
    #for w in vocabulary:
        #for s in suffixes:
            #analysis = w + s
            #output = FSM.runFST([fsa],[analysis])[0][0]
            #if output != analysis:
                #print "FAILED"
                #sys.exit(1)

    print "==== Generating random paths for 'aced', using only channel model ===="
    output = FSM.runFST([fst], ["aced"], maxNumPaths=10, randomPaths=True)
    print "==== Result: ", str(output), " ===="

    print "==== Generating random paths for 'aced', using only channel model ===="
    output = FSM.runFST([fst], ["panicked"], maxNumPaths=10, randomPaths=True)
    print "==== Result: ", str(output), " ===="

    print "==== Disambiguating a few phrases: aced, panicked, paniced, sprucing ===="
    output = FSM.runFST([fsa,fst], ["aced", "paniced", "panicked", "sprucing"])
    print "==== Result: ", str(output), " ===="

if __name__ == '__main__':
    simpleTest()
