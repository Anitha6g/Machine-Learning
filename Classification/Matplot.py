import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
x = range(10)
plt.plot(x,cv_scores)
plt.plot(x,cv_scores1)
plt.xlabel('Range of 10-fold values(1-10)')
plt.ylabel('Score of 10-fold models')
KNN_Alg = mpatches.Patch(color='blue', label='KNN')
NB = mpatches.Patch(color='red', label='Naive Bayes')
plt.legend(handles=[KNN_Alg,NB])