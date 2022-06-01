import os
import pathlib
import numpy as np

rootDir = pathlib.Path('/home/brian.atkinson/thesis/data/noGPU/hybrid')
humanScores,baseScores = [],[]
bestBaseScore,bestBaseDir,bestHumScore,bestHumDir  = 0, None,0, None
worstBaseScore,worstBaseDir,worstHumScore,worstHumDir = 100, None,100, None
hybDict = {'.25':{'best':0,'worst':100,'bestDir':None,'worstDir':None,'scores':[]},
            '.5':{'best':0,'worst':100,'bestDir':None,'worstDir':None,'scores':[]},
            '.99':{'best':0,'worst':100,'bestDir':None,'worstDir':None,'scores':[]}
    }
weightDict = {'Xavier':{'best':0,'worst':100,'bestDir':None,'worstDir':None,'scores':[]},
            'He':{'best':0,'worst':100,'bestDir':None,'worstDir':None,'scores':[]}
    }
activDict = {'Leaky ReLu':{'best':0,'worst':100,'bestDir':None,'worstDir':None,'scores':[]},
                'ReLu':{'best':0,'worst':100,'bestDir':None,'worstDir':None,'scores':[]}
    }
for exp in rootDir.iterdir():
    if True:
    # for exp in d.iterdir():
        # try:
        if True:
            with open(os.path.join(exp,'digest.txt'),'r') as fd:
                lines = fd.readlines()
                best = int(lines[1].split(',')[1].split(':')[1])
                print(exp,best)
            if exp.name[0]=='n':
                baseScores.append(best)
                if best > bestBaseScore:
                    bestBaseDir = exp
                    bestBaseScore = best
                if best < worstBaseScore:
                    worstBaseDir = exp
                    worstBaseScore = best
            else:
                humanScores.append(best)
                if best > bestHumScore:
                    bestHumDir = exp
                    bestHumScore = best
                if best < worstHumScore:
                    worstHumDir = exp
                    worstHumScore = best
            partString = str(exp)
            parts = partString.split('/')[-1].split('-')
            for part in parts:
                try:
                    if 'Hyb' in part:
                        key = part.split('0')[1]
                        if best > hybDict[key]['best']:
                            hybDict[key]['best'] = best
                            hybDict[key]['bestDir'] = exp
                        if best < hybDict[key]['worst']:
                            hybDict[key]['worst'] = best
                            hybDict[key]['worstDir'] = exp
                        hybDict[key]['scores'].append(best)
                except KeyError:
                    pass
                if 'Init' in part:
                    key = part.split('_')[1]
                    if best > weightDict[key]['best']:
                        weightDict[key]['best'] = best
                        weightDict[key]['bestDir'] = exp
                    if best < weightDict[key]['worst']:
                        weightDict[key]['worst'] = best
                        weightDict[key]['worstDir'] = exp
                    weightDict[key]['scores'].append(best)
                if 'Leaky' in part:
                    condition = part.split('_')[1]
                    key = 'Leaky ReLu' if condition=='True' else 'ReLu'
                    if best > activDict[key]['best']:
                        activDict[key]['best'] = best
                        activDict[key]['bestDir'] = exp
                    if best < activDict[key]['worst']:
                        activDict[key]['worst'] = best
                        activDict[key]['worstDir'] = exp
                    activDict[key]['scores'].append(best)
humanArray = np.array(humanScores)
baseArray = np.array(baseScores)


names = {'hybridResults.txt':hybDict,'weightInitResults.txt':activDict,'weightInitResults.txt':weightDict}

for name in names.keys():
    d= names[name]
    with open(name,'w') as fd:
        fd.write('human avg:{}    base avg:{}\n'.format(np.mean(humanArray),np.mean(baseArray)))
        for k in d.keys():
            fd.write('\n{} Stats\n'.format(k))
            scoreArray = np.array(d[k]['scores'])
            fd.write('avg:{}   median:{}\n'.format(np.mean(scoreArray),np.median(scoreArray)))
            for k2 in d[k].keys():
                if k2 != 'scores':
                    fd.write('{}:{}\n'.format(k2,d[k][k2]))