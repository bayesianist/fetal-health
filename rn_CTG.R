
#construire l'architecture du perceptron ???
#parametrage de l'algo d'apprentissage ???



#Reseau de neuronnes avec une couche cachee

#source : http://eric.univ-lyon2.fr/~ricco/cours/slides/reseaux_neurones_perceptron.pdf
#https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/

#Probleme : Cardiotocogrammes du portugal
#detecter une souffrance foetale (response a K =3 niveaux, normal suspet patho)
#à partir de la description du CTG  
#avec n = 2126 et p=21

#install.packages("nnet")
library(neuralnet)

#probleme a deux classes : 1 couche cachee a 2 neurones 
#codage explicite de la cible ! - si codage 1/0 = l'algorithme ne converge pas
#correction des coeff se font mal durant l'apprentissage

#sources : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Tensorflow_Keras_R.pdf


######################### 1. CODAGE EXPLICITE

#eviter la saturation : quand les valeurs des combinaisons lineaires sont elevees en VA
#et que la transformation sigmoide sature en 0 ou 1 (zone ou derivee nulle)

#codage de la cible 0.8:p  0.2:n
NSP.bin <-ifelse(ctg.cr$NSP =="p",1,0)
print(table(NSP.bin))
#binariser la variable qualitative (-1:patho, 1 ou 0:non patho)
Tendency.bin <- as.factor(ifelse(ctg.cr$Tendency =="-1",1,0))
print(table(Tendency.bin))
#data binarises
ctg.cr.bin <- cbind(ctg.cr[,1:20], Tendency.bin,  NSP.bin)


######################### 2. PARAMETRAGE

#parametrer un perceptron multicouches ?
#source : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Optimal_Neurons_Perceptron.pdf

#formule - liste des explicatives
formule <- paste(colnames(ctg.cr[,1:20]), collapse = "+")
formule <- paste(formule, "+", "Tendency.bin",sep="")
print(formule)
#suite formule, associer variable cible
formule <- paste("NSP.bin~", formule, sep="")
print(formule)



######################## 3. APPRENTISSAGE et PREDICTION en CV

#re-echantillonnage par CV pour estimer les parametres

#sources : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Validation_Croisee_Suite.pdf 

#methode ricco

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

#apprentissage en CV

#sources : http://tutoriels-data-mining.blogspot.com/search?q=neuralnet 

#lancer la validation croisee
 all.err <- numeric(0)
 for (k in 1:K){
   #apprendre le modele sur tous les individus sauf le bloc k
   set.seed(100)
   pm.neural <- neuralnet::neuralnet(as.formula(formule),
                                     data=data.matrix(ctg.cr.bin[bloc !=k,]),
                          hidden=c(2),linear.output=FALSE)
   #appliquer le modele sur le bloc numero k
   proba.pred.pm.neural <- predict(pm.neural, data.matrix(ctg.cr.bin[bloc==k,-ncol(ctg.cr.bin)]))
   #traduire les modalites
   test.mc <- ifelse(ctg.cr.bin[bloc==k,NSP.bin]>0.5,"p","n")
   pred.pm.neural <- ifelse(proba.pred.pm.neural>0.5,"p","n")
   #matrice de confusion
   mc <-table(test.mc[bloc==k,NSP.bin],pred.pm.neural)
   #taux d'erreur
   err <-1-sum(diag(mc))/sum(mc)
   #conserver taux d'erreur
   all.err <- rbind(all.err,err)
 }

 #afficher les poids calcules via graphe
 plot(pred_sc_cv$net.result)
 plot(nnet_fit_1hl_cv)
 
 #vecteur des erreurs rate recueillies
 print(all.err)
 mean(all.err) #29%  
 
#methode JAP 
# 
# # tirage au sort k échantillons de taille n/k
# nb_folds = 10
# folds_obs = sample(rep(1:nb_folds,length.out=n))
# 
# for (k in 1:nb_folds){
#   print (paste("=====> Fold :", k))
#   #distinguer les individus test
#   test = which(folds_obs == k) 
#   # des individus d'entrainement
#   train = setdiff (1:n, test)
#   #apprentissage reseau de neuronne a une couche cachee
#   nnet_fit_1hl_cv=neuralnet(formula=as.formula(formule),
#                             data=data.matrix(ctg.cr.bin[train,]),
#                             hidden=c(2),
#                             # err.fct="ce",
#                             linear.output=FALSE)
#   #test
#   pred_sc_cv = compute(nnet_fit_1hl_cv,data.matrix(ctg.cr.bin[test,-22]))
#   #erreur de test du fold error_test
#   colnames(pred_sc_cv$net.result) <- "ctg_pathologiques" #renommer les variables des resultats 
#   ## matrice de confusion
#   test_mat <- cbind(ctg.cr.bin[test,0.8],pred_sc_cv$net.result)
# }
# # mean(Y[test,] != pred[test,nom_class_cib])

####################### 6. EVALUATION
