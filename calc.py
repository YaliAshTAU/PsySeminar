Human= []
Model = []

TP, FP, FN, TN = 0, 0, 0, 0
for i in range(len(Model)):
        if Model[i] == True and Human[i] == True:
            TP += 1
        if Model[i] == True and Human[i] == False:
            FP += 1
        if Model[i] == False and Human[i] == True:
            FN += 1
        if Model[i] == False and Human[i] == False:
            TN += 1

print('TP:', TP)
print('FP:', FP)
print('FN:', FN)
print('TN:', TN)

print('Accuracy:', (TP + TN) / (TP + TN + FP + FN))
print('Precision:', TP / (TP + FP))
print('Recall:', TP / (TP + FN))
print('F1 Score:', 2 * (TP / (TP + FP) * TP / (TP + FN)) / (TP / (TP + FP) + TP / (TP + FN)))

