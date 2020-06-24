#Reseau de neuronnes avec une couche cachee

#install.packages("nnet")
library(neuralnet)

#source : http://eric.univ-lyon2.fr/~ricco/cours/slides/reseaux_neurones_perceptron.pdf
#http://penseeartificielle.fr/focus-reseau-neurones-artificiels-perceptron-multicouche/

#Probleme : Cardiotocogrammes du portugal
#detecter une souffrance foetale (response a K =2 niveaux, normal ou suspet-patho)
#a partir de la description du CTG avec n = 2126 et p=21
#codage explicite de la cible ! - si codage 1/0 = l'algorithme ne converge pas
#correction des coeff se font mal durant l'apprentissage

#sources : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Tensorflow_Keras_R.pdf


############################################# I APPRENTISSAGE et PREDICTION en CV

#formule - liste des explicatives
formule <- paste(colnames(ctg.cr.bin[,1:20]), collapse = "+")
formule <- paste(formule, "+", "Tendency.bin",sep="")
print(formule)
#suite formule, associer variable cible
formule <- paste("NSP.bin~", formule, sep="")
print(formule)

#re-echantillonnage par CV pour estimer les parametres
# http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Validation_Croisee_Suite.pdf 

#determiner le numero de bloc de chaque individu
n <- nrow(ctg.cr.bin)
K <- 10
taille <- n%/%K
set.seed(5)
alea <-runif(n)
rang <- rank(alea)
bloc <- (rang-1)%/%taille+1
bloc <- as.factor(bloc)
print(summary(bloc)) #212 (10 blocs) + 1 de 6

#fonction pour une CV a partir de RN avec plusieurs neuronnes (2 par defaut) de la couche cachee
#cross entropy car classement
#logistic car tanh ne marche pas

crossvalidate.NN <- function(hidden=c(2)){
  #initialiser
  all.rappel <-numeric(0)
  all.precision <-numeric(0)
  all.F1 <-numeric(0)
  all.err <-numeric(0)
  #faire la CV sur les k folds
   for (k in 1:K){
   #apprendre le modele sur tous les individus sauf ceux du bloc k
   pm.neural <- neuralnet::neuralnet(as.formula(formule), data=data.matrix(ctg.cr.bin[bloc !=k,]), act.fct = "logistic", hidden=hidden, threshold=0.01,
                                     linear.output=FALSE)
   #tester sur le bloc numero k
   proba.pred.pm.neural <- predict(pm.neural, data.matrix(ctg.cr.bin[bloc==k,-ncol(ctg.cr.bin)]))
   #traduire les modalites
   test.mc <- ifelse(ctg.cr.bin[bloc==k,"NSP.bin"]>0.5,"p","n")
   pred.pm.neural <- ifelse(proba.pred.pm.neural>0.5,"p","n")
   #afficher les parametres de performance de chaque fold
   print(paste("----------------------------------------------> fold :",k))
   #matrice de confusion
   mc <-table(pred.pm.neural,test.mc)
   print("Matrice de confusion :"); print(mc)
   #taux d'erreur
   err <-1-sum(diag(mc))/sum(mc)
   print(paste("Taux d'erreur =", round(err,3)))
   #conserver taux d'erreur
   all.err <- rbind(all.err,err)
   #rappel 
   recall <- mc["p","p"]/sum(mc[,"p"]) 
   print(paste("Rappel =", round(recall,3)))
   #conserver le rappel
   all.rappel <- rbind(all.rappel,recall)
   #precision 
   precision <- mc["p","p"]/sum(mc["p",]) 
   print(paste("Precision =",round(precision,3)))
   #conserver precision
   all.precision <- rbind(all.precision,precision)
   #F1-Measure 
   f1 <- 2.0*(precision*recall)/(precision+recall) 
   print(paste("F1-Score =",round(f1,3)))
   #conserver F1
   all.F1 <- rbind(all.F1,f1)
 }

 #graphe du RN
 plot(pm.neural)
 #vecteur des erreurs rate recueillies
 print(all.err)
 mean(all.err) #8,3%
 #moyenne des precision / VPP
 print(all.precision)
 mean(all.precision) #80% CTG correctement diagnostic patho / nb CTG diagnostic patho
 #moyenne des rappels / sensibilité
 print(all.rappel)
 mean(all.rappel) #83% CTG correctement diagnostic patho / nb CTG patho
 #moyenne des parametres de performance
 print(all.F1)
 return(round(mean(all.F1), digits=3)) #81%
}




###################################################################### II TUNING

########################## faire tourner la fonction de CV pour estimer les parametres :

library(MASS)
library(neuralnet)
library(plyr)

#Pour la categorisation en q=2 classes : un seul perceptron de sortie et la fonction sigmoide
#sources : cm de JAP, http://tutoriels-data-mining.blogspot.com/search?q=neuralnet
#https://www.r-bloggers.com/selecting-the-number-of-neurons-in-the-hidden-layer-of-a-neural-network/
#https://meritis.fr/ia/deep-learning/
#http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Optimal_Neurons_Perceptron.pdf

#tester les 4 seuils pour les 6 modeles de RN (5 à 10 neuronnes dans la couche cachée) :

cv.tunin.threshold<-c(0)
vec <- c(0.001,0.01,0.1,0.5)
for (i in 1:length(vec)) {
  threshold <- vec[i]
  print(paste("===========================================> threshold :",i))
  resultat.tunin.threshold <- sapply(c(5:10),crossvalidate.NN)
  cv.tunin.threshold <- cbind(cv.tunin.threshold,resultat.tunin.threshold)
}


########################################################## GRAPHIQUES COMPARATIFS

#nombre de neuronnes dans la couche cachée
all.resultats<-c(0)
for (i in 1:length(vec)) {
  vec <- c(0.001,0.01,0.1,0.5)
  seuil <- vec[i]
  print(paste("===========================================> threshold :",i))
  #moyenne des F1 pour 5 à 10 neuronnes dans la couche cachée
  resultats.hiddenl.tuning <- sapply(c(5:10),crossvalidate.NN)
  all.resultats <- cbind(all.resultats,resultats.hiddenl.tuning)
}

nb.neuronnes <- c(5:10)
resultats.hiddenl <- cbind(nb.neuronnes,all.resultats[,-1]) #5 neuronnes
resultat.table <- as.data.frame(resultats.hiddenl)#pour threshold à 0.01 et hidden.n à 9

plot(nb.neuronnes,resultats.hiddenl.tuning, type = "b", 
     xlab = "nombre de neuronnes dans la couche cachée", ylab = "F1-mesure")

