#orienter adroitement le paramètre de régularisation après avoir défini 
#un critère d’évaluation calculé sur un fichier test

#source : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Optimal_Neurons_Perceptron.pdf

#parametres : SVM-type, VSM-Kernal, cost, gamma, nu


#prediction

nusvr1.pred <- predict(nu.svr, newdata = donnees.test)
nusvr1.rss <- sum((donnees.test$NSP - nusvr1.pred)^2)

# Pseudo‐R2

print(1.0-nusvr1.rss/def.rss)

