import csv
import matplotlib.pyplot as plt

# dir = 'heuristic_test/ht-500000-S42-loss-5.0-hum0.4/'
dir = 'heuristic_test/no_ht-500000-S42-loss-5.0/'
eps=[]
scores=[]
with open(dir+'stats.csv',newline='') as csvFile:
    reader = csv.reader(csvFile,delimiter=',')
    for line in reader:
        ep,score=int(line[0]),int(line[1])
        eps.append(ep)
        scores.append(score)
plt.scatter(eps,scores,marker='.')
plt.savefig(dir+'learning.png')