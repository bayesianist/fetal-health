#comparer les performances de deux algorithmes d’apprentissage supervisé
#à l’aide de l’erreur calculée en validation croisée.

# source : https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/

#estimation de la MSE et son IC
boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)

#AUC