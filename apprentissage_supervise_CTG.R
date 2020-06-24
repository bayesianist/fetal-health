# Rappel de la problématique : Cardiotocogrammes (CTG) des foetus a la maternite,
#detecter une souffrance foetale NSP = "p" (patho ou suspect)
#a partir des caracteristiques (p=21) du trace des CTG (n= 2126)

#NGUYEN Long & ANSOBORLO Marie

# ###################################################################
#                                                                   #
#    1.                                                             #
#                                                                   #
#          PRE TRAITEMENT DES DONNES DE CTG                         #
#                                                                   #     
# ###################################################################

library(dplyr)
library(xlsx)
library(XLConnect)
library(readr)

#importer les data bruts
cardiotocogram <- read.xlsx("CTG.xls", sheetIndex = 3)
dim(cardiotocogram) # taille de la base : 2126 individus, 42 variables

#selection des variables d'interet
descri <- cardiotocogram[,10:30]
ctg <- cbind(descri,NSP = cardiotocogram[,"NSP"]) # nb var (22) : 21 desc + NSP (Y)
dim(ctg) #taille de notre jeu de donnee : 2126 individus, 22 variables
summary(ctg)

#mettre en factor le descripteur tendency
ctg$Tendency <- as.factor(ctg$Tendency)
ctg$NSP <- as.factor(ctg$NSP)
lapply(ctg, class) #verifier les classes : 20 de type num et 2 facteur

#recoder la variable a predire : patho (2 et 3) ou non (1)
ctg$NSP <- factor(ifelse(ctg$NSP=="1","n","p"))
print(table(ctg$NSP)) #1655 CTG normaux et 471 suspects/patho


############################################## ECHANTILLONS POUR L'APPRENTISSAGE, TEST

library(caret)

# subdivision en apprentissage (70%) et test (30%), validation set approach

#fixe la graine du tirage aleatoire
set.seed(2) 
training.idx <- createDataPartition(ctg$NSP, p=0.7, list=FALSE) #stratifier sur Y (NSP)

#creer le jeu d'apprentissage
my_train <- ctg[training.idx,]
dim(my_train) # taille de l'échantillon d'apprentissage : 1489 obs
round(table(my_train$NSP)/nrow(my_train),2) # 78% sont sains, 22% patho (tx erreur theorique)

#creer le jeu de test
my_test <- ctg[-training.idx,]
dim(my_test) # taille de l'échantillon de test : 637 obs
round(table(my_test$NSP)/nrow(my_test),2) # 78% sains et 22 pathos

#separer cible et descripteurs
Xtrain = my_train[,-22]
ytrain = my_train[,22]
Xtest = my_test[,-22]
ytest = my_test[,22]


################################################################ CENTRER ET REDUIRE LES VA num

# /!\ Seuls les parametres calcules sur le train doivent intervenir dans la trasnformation des  donnees.
# utiliser exclusivement les moyennes et ecarts-type calcules sur “train” pour normaliser les deux echantillons.
#source : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Packages_R_for_Deep_Learning.pdf 

# calcul des moyennes sur l'echantillon d'apprentissage
mean.train <- sapply(as.data.frame(Xtrain[,1:20]),mean)
print(mean.train)

#idem pour l'ecart-type
sd.train <- sapply(as.data.frame(Xtrain[,1:20]),sd)
print(sd.train)

#centrage-reduction de l echantillon d apprentissage
ctg.cr21 <- data.frame(scale(as.data.frame(ctg[,1:20]),
                             center=mean.train,
                             scale=sd.train))

#verifier
print(colMeans(ctg.cr21))

#rajouter VA factorielles au dataset
Tendency <- ctg$Tendency
NSP <- ctg$NSP
ctg.cr <- cbind(ctg.cr21,Tendency,NSP)
summary(ctg.cr)

#echantillons centres et reduits
Xtrain.cr = ctg.cr[training.idx,-22]
Xtest.cr = ctg.cr[-training.idx,-22]
train.cr = ctg.cr[training.idx,]
test.cr = ctg.cr[-training.idx,]


########################################################## CODAGE EXPLICITE

#eviter la non convergence du modele pour la regression elasctinet
#la transformation sigmoide sature en 0 ou 1 (zone ou derivee nulle) 
#quand les valeurs des combinaisons lineaires sont elevees

#codage de la cible 0.8:p  0.2:n
NSP.bin <-ifelse(ctg.cr$NSP =="p",0.8,0.2)
NSP.bin2 <-ifelse(ctg.cr$NSP =="p",1,0)
print(table(NSP.bin))
#binariser la variable qualitative (-1:patho, 1 ou 0:non patho)
Tendency.bin <- as.factor(ifelse(ctg.cr$Tendency =="-1",1,0))
print(table(Tendency.bin))
#data binarises
ctg.cr.bin <- cbind(ctg.cr[,1:20], Tendency.bin,  NSP.bin)
ctg.cr.bin2 <- cbind(ctg.cr[,1:20], Tendency.bin,  NSP.bin2)

#séparer les échantillons
train.cr.bin = ctg.cr.bin[training.idx,]
test.cr.bin = ctg.cr.bin[-training.idx,]
train.cr.bin2 = ctg.cr.bin2[training.idx,]
test.cr.bin2 = ctg.cr.bin2[training.idx,]


# ###################################################################
#                                                                   #
#      2.                                                           #
#          REGRESSION PENALISEE PAR UNE FONCTION DE REGULARISATION  #
#                                                                   #     
# ###################################################################

#sources : http://eric.univ-lyon2.fr/~ricco/cours/slides/regularized_regression.pdf
#http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Ridge_Elasticnet_R.pdf
#https://mrmint.fr/initiation-r-machine-learning

#install.packages("glmnetUtils")
library(rpart)
library(glmnet)
library(glmnetUtils)

############################ 1. APPRENTISSAGE, ESTIMER LES PARAMETRES DU MODELE

#source : https://cran.r-project.org/web/packages/glmnetUtils/vignettes/intro.html
#source : https://stackoverflow.com/questions/54803990/extract-the-best-parameters-from-cva-glmnet-object

library(data.table)
library(glmnetUtils)
library(useful)

#CV for alpha : optimisation conjointe de lambda (shrinkage) et alpha (mixing)

### fit  (/!\ glmnet prend que matrix) pour 5 modeles differents selon 5 alpha
set.seed(100)
cva.enet <- cva.glmnet(x=data.matrix(Xtrain.cr), y=data.matrix(ytrain),
                       nfolds=10, alpha=c(0.01, 0.1, 0.5, 0.8, 1.0), family="binomial", standardize=FALSE)
plot(cva.enet, xvar="lambda") #alpha 1 minimise la binom dev


### extraire mse, lambda et alpha
nb.of.alpha.testes <- length(cva.enet$alpha)
cva.glmnet.dt <- data.table()
#boucler sur chaque alpha
for (i in 1:nb.of.alpha.testes){
  glmnet.model <- cva.enet$modlist[[i]]
  min.mse <-  min(glmnet.model$cvm) #mean cross validated error
  min.lambda <- glmnet.model$lambda.min #plus faible shrinkage et parcimonieux
  alpha.value <- cva.enet$alpha[i] 
  new.cva.glmnet.dt <- data.table(alpha=alpha.value, min_mse=min.mse, min_lambda=min.lambda)
  cva.glmnet.dt <- rbind(cva.glmnet.dt,new.cva.glmnet.dt)
}

#recupere l'alpha pour la plus faible mse
best.params <- cva.glmnet.dt[which.min(cva.glmnet.dt$min_mse)] 
print(best.params) #alpha 1, lambda 0.001--> MODEL LIST : 5
# CCL besoin d'une régularisation tres faible et d'un arbitrage plus lasso.
#α est tres eleve, on impose beaucoup de contraintes sur les coefficients 
#danger du sous-apprentissage si alpha :1, ne plus exploiter efficacement les donnees)

#afficher nb de variables selectionnes vs lambda (alpha 1)
cbind(cva.enet$modlist[[3]][["lambda"]],cva.enet$modlist[[4]][["nzero"]])
# 20 variables donc 1 exclue

#coeff estimés des variables pour le meilleur modele (sur var centree et reduites)
coef(cva.enet$modlist[[3]],s="lambda.min") #Width a coeff nul


############################### 2. PREDICTION SUR TEST, PARAMETRES PERFORMANCE

#fonction calcul de la F1 mesure
calc.f1 <- function(y,predict){
  mc <-table(y ,predict)
  recall <- mc["p","p"]/sum(mc[,"p"]) 
  precision <- mc["p","p"]/sum(mc["p",]) 
  f1 <- 2.0*(precision*recall)/(precision+recall)
  return(f1)
}

# predictions selon le lambda et alpha 1
#cva lambda 0.00112
yenet3 <- predict(cva.enet$modlist[[5]], data.matrix(Xtest.cr),
                  s=c(cva.enet$modlist[[5]][["lambda.min"]]), type = "class")
fl001 <- calc.f1(test.cr$NSP,yenet3) #76.5%
#lambda 0.0112
yenetl01 <- predict(cva.enet$modlist[[5]], data.matrix(Xtest.cr),
                    s=c(0.0112), type = "class")
fl010 <- calc.f1(test.cr$NSP,yenetl0.01) #76.8%
#lambda 0.016 
yenetl016 <- predict(cva.enet$modlist[[5]], data.matrix(Xtest.cr),
                     s=c(0.016), type = "class")
fl016 <- calc.f1(test.cr$NSP,yenetl0.016)#75.7%
#lambda 0.112 
yenetl025 <- predict(cva.enet$modlist[[5]], data.matrix(Xtest.cr),
                     s=c(0.112), type = "class")
fl025 <- calc.f1(test.cr$NSP,yenetl0.025)#21.5%


############################################ 3. GRAPHIQUES DE COMPARAISON

#ALPHA : tracer la binomiale deviance selon differents alpha utilises (ridge au lasso)
par(mfrow=c(2,2))
plot(cva.enet$modlist[[1]]);plot(cva.enet$modlist[[3]]);plot(cva.enet$modlist[[5]])

plot(log(cva.enet$modlist[[1]]$lambda),cva.enet$modlist[[1]]$cvm,pch=19,col="red",xlab="log(Lambda)",ylab=cva.enet$modlist[[1]]$name)
points(log(cva.enet$modlist[[3]]$lambda),cva.enet$modlist[[3]]$cvm,pch=19,col="blue")
points(log(cva.enet$modlist[[5]]$lambda),cva.enet$modlist[[5]]$cvm,pch=19,col="orange")
legend("topright",legend=c("alpha= 0.1","alpha= 0.5","alpha= 1.0"),
       pch=19,col=c("red","blue","orange"),cex=0.15,fill=1:6)


#LAMBDA : valeurs pour l'histogramme des différents lambda du lasso

vec.mod.lamb<- round(as.data.frame(cbind(fl001,fl010,fl016,fl025)),3)
table.lamb <- t(vec.mod.lamb)
vec.mod.lamb.f1 <- c("0.001","0.01","0.016","0.025")
table.lamb<- cbind(table.lamb,vec.mod.lamb.f1)
colnames(table.lamb) <- c("F1","lambda.lasso")

#tracer la courbe

ggplot(as.data.frame(table.lamb))+
  aes(x= lambda.lasso, y = F1, fill=lambda.lasso)+
  geom_bar(stat="identity")+
  xlab("lambda (pour lasso)")+
  ylab("F1 mesure")+
  ggtitle("F1 mesure selon les lambda (pour un modele type lasso)")



################################################ 4. TABLEAUX DE COMPARAISON des F1

# predictions selon les alphas sur les memes donnees test
# plot(cva.enet$modlist[[1]]) #alpha 0.01 lambda min
# yenet1 <- predict(cva.enet$modlist[[1]], data.matrix(Xtest.cr),
#                   s=c(cva.enet$modlist[[1]][["lambda.min"]]), type = "class")
# mse1 <- mean(yenet1 != ytest) #10.2%

plot(cva.enet$modlist[[1]]) #alpha 0.01 lambda min 0.01
yenet1 <- predict(cva.enet$modlist[[1]], data.matrix(Xtest.cr),
                  s=c(0.01), type = "class")
f1.1 <- calc.f1(yenet1,ytest) #76.9%

plot(cva.enet$modlist[[2]]) #alpha 0.1 lambda min 0.01
yenet2 <- predict(cva.enet$modlist[[2]], data.matrix(Xtest.cr),
                  s=c(0.01), type = "class")
f1.2 <- calc.f1(yenet2,ytest) #77.4%

plot(cva.enet$modlist[[4]]) #alpha 0.8 lambda min 0.01
yenet4 <- predict(cva.enet$modlist[[4]], data.matrix(Xtest.cr),
                  s=c(0.01), type = "class")
f1.4 <- calc.f1(yenet4,ytest) #76.7%

plot(cva.enet$modlist[[5]]) #alpha 1.0 lambda min 0.01
yenet5 <- predict(cva.enet$modlist[[5]], data.matrix(Xtest.cr),
                  s=c(0.01), type = "class")
f1.5 <- calc.f1(yenet5,ytest) #76.5%

plot(cva.enet$modlist[[3]]) #alpha 0.5 lambda min 0.01
yenet3 <- predict(cva.enet$modlist[[3]], data.matrix(Xtest.cr),
                  s=c(0.01), type = "class")
f1.3 <- calc.f1(yenet3,ytest) #75.7%

#graphiques des alpha différents

vec.mod.alpha<- round(as.data.frame(cbind(f1.1,f1.2,f1.3,f1.4,f1.5)),3)
table.alpha <- t(vec.mod.alpha)
vec.mod.alpha.f1 <- c("0.01","0.1","0.5","0.8","1.0")
table.alpha <- cbind(table.alpha,vec.mod.alpha.f1)
colnames(table.alpha) <- c("F1","alpha")

#tracer la courbe

ggplot(as.data.frame(table.alpha))+
  aes(x= alpha, y = F1, fill=alpha)+
  geom_bar(stat="identity")+
  xlab("alpha (pour lambda 0.01)")+
  ylab("F1 mesure")+
  ggtitle("F1 mesure selon les alpha (pour un shrinkage faible)")



# ###################################################################
#                                                                   #
#     3.                                                            #
#          RESEAU DE NEURONNES A UNE COUCHE CACHEE                  #
#                                                                   #     
# ###################################################################


library(nnet)
library(neuralnet)

#source : http://eric.univ-lyon2.fr/~ricco/cours/slides/reseaux_neurones_perceptron.pdf
#http://penseeartificielle.fr/focus-reseau-neurones-artificiels-perceptron-multicouche/
#http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Tensorflow_Keras_R.pdf


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

mean(all.resultats)

nb.neuronnes <- c(5:10)
resultats.hiddenl <- cbind(nb.neuronnes,all.resultats[,-1]) #5 neuronnes
resultat.table <- as.data.frame(resultats.hiddenl)#pour threshold à 0.01 et hidden.n à 9

plot(nb.neuronnes,resultats.hiddenl.tuning, type = "b", 
     xlab = "nombre de neuronnes dans la couche cachée", ylab = "F1-mesure")




# ###################################################################
#                                                                   #
#     4.                                                            #
#          SEPARATEUR A VASTES MARGES                               #
#                                                                   #     
# ###################################################################

#source : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Optimal_Neurons_Perceptron.pdf
#http://eric.univ-lyon2.fr/~ricco/cours/slides/svm.pdf

library(e1071)


############################################################## SVM TUNE

#SVM par defaut
mod.def <- svm(my_train$NSP~., data= my_train, scale= TRUE) 
print(attributes(mod.def)) #gamma 0.04347826, cost=1, kernel = radial

# grille de recherche des meilleurs parametres en CV : radial et cost 10
set.seed(1000) #pour la reproductibilite
obj1 <- tune(svm, NSP~., data = my_train, ranges =
               list(kernel=c('linear','polynomial','radial','sigmoid'),
                    cost = c(0.1, 0.5, 1.0, 2.0, 10),
                    tunecontrol = tune.control(sampling="cross")))



#################################### EVALUATION

#install.packages("ggplot2")
library(ggplot2)
library(questionr)
library(ggpirate)

################### FONCTION POUR ENTRAINER EN CV

#determiner le numero de bloc de chaque individu
n <- nrow(my_train)
K <- 10
taille <- n%/%K
set.seed(5)
alea <-runif(n)
rang <- rank(alea)
bloc <- (rang-1)%/%taille+1
bloc <- as.factor(bloc)
print(summary(bloc)) #10 blocs de 148 individus + 1 de 9

########## 

crossvalidate.SVM <- function(kernel='radial',cost=1, gamma=0.04){
  
  #initialiser les vecteurs
  all.rappel <- numeric(0)
  all.precision <- numeric(0)
  all.F1 <- numeric(0)
  all.err <- numeric(0)
  
  for (k in 1:K){
    #apprendre le modele sur tous les individus sauf le bloc k
    mod.kernel.cost<-svm(NSP~.,data=ctg.cr[bloc!=k,],kernel=kernel,gamma=gamma, cost=cost,scale=TRUE)
    #test du modele sur le bloc numero k
    predict <- predict(mod.kernel.cost, newdata = ctg.cr[bloc==k,])
    #PARAMETRES DE PERFORMANCE par CV
    print(paste("----------------------------------------------> fold :",k))
    #matrice de confusion
    mc <-table(ctg.cr[bloc==k,"NSP"],predict)
    print("Matrice de confusion :"); print(mc)
    #taux d'erreur
    err <-1-sum(diag(mc))/sum(mc)
    print(paste("Taux d'erreur =", round(err,3)))
    #conserver taux d'erreur
    all.err <- rbind(all.err,err)
    #rappel 
    recall <- mc["p","p"]/sum(mc[,"p"]) 
    print(paste("Rappel =", round(recall,3)))
    all.rappel <- rbind(all.rappel,recall)
    #precision 
    precision <- mc["p","p"]/sum(mc["p",]) 
    print(paste("Precision =",round(precision,3)))
    all.precision <- rbind(all.precision,precision)
    #F1-Measure 
    f1 <- 2.0*(precision*recall)/(precision+recall) 
    print(paste("F1-Score =",round(f1,3)))
    all.F1 <- rbind(all.F1,f1)
  }
  
  #vecteur des erreurs rate recueillies
  print(all.err)
  mean(all.err) #8,3%
  #moyenne des precisions / VPP
  print(all.precision)
  mean(all.precision) #80% CTG correctement diagnostic patho / nb CTG diagnostic patho
  #moyenne des rappels / sensibilité
  print(all.rappel)
  mean(all.rappel) #83% CTG correctement diagnostic patho / nb CTG patho
  #moyenne des parametres de performance
  print("toutes les F1 par fold :")
  print(all.F1)
  return(round(mean(all.F1),digits = 3)) #81%
}

############################### EVALUATION SELON LES KERNEL

mod.rad10 <- crossvalidate.SVM(kernel='radial',cost=10) #5.5% taux erreur
mod.lin <- crossvalidate.SVM(kernel='linear',cost=10) #8.9% taux erreur
mod.pol <- crossvalidate.SVM(kernel='polynomial',cost=10) #6.4% taux erreur
mod.sig <- crossvalidate.SVM(kernel='sigmoid',cost=10) #21% taux erreur

#diagramme en barre des F1 selon les modeles

#source : http://www.sthda.com/french/wiki/ggplot2-barplots-guide-de-demarrage-rapide-logiciel-r-et-visualisation-de-donnees

vec.mod.cost10.f1 <- as.data.frame(cbind(mod.rad10[1],mod.lin[1],mod.pol[1],mod.sig[1]))
table.cost10.f1 <- t(vec.mod.cost10.f1)
vec.mod.cost10.f1 <- c("radial","lineaire","polynomial","sigmoide")
table.cost10.f1 <- cbind(table.cost10.f1,vec.mod.cost10.f1)
colnames(table.cost10.f1) <- c("F1","Kernel")

ggplot(as.data.frame(table.cost10.f1))+
  aes(x= Kernel, y = F1, color=F1)+
  geom_bar(stat="identity", fill="white")+
  xlab("fonction kernel")+
  ylab("F1 mesure")+
  ggtitle("F1 mesure selon les fonctions kernel (pour un cost:10)")



############################# EVALUATION SELON LES COST

#cout de la violation de la contrainte dans la relaxation lagrangienne
#the greater the cost parameter, the more are penalize the residuals resulting in higher variance and lower bias.

mod.def <- crossvalidate.SVM(kernel='radial',cost=1) #83.4% F1
mod.rad50 <- crossvalidate.SVM(kernel='radial',cost=50) #87.5% 
mod.rad100 <- crossvalidate.SVM(kernel='radial',cost=100) #87.8% 
mod.rad150 <- crossvalidate.SVM(kernel='radial',cost=150) #86.7% 

#diagramme en barre des F1 selon les modeles

library(forcats)

#source : http://www.sthda.com/french/wiki/ggplot2-barplots-guide-de-demarrage-rapide-logiciel-r-et-visualisation-de-donnees
#http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_SVM_R_Python.pdf 

vec.mod.rad.f1 <- as.data.frame(cbind(mod.def[1],mod.rad10[1],mod.rad50[1],mod.rad100[1],mod.rad150[1]))
table.rad.f1 <- t(vec.mod.rad.f1)
vec.mod.rad.f1 <- c("1","10","50","100","150")
# vec.mod.rad.f1 <- ordered(vec.mod.rad.f1, levels = c("1","10","50",'100',"150"))
table.rad.f1 <- cbind(table.rad.f1,vec.mod.rad.f1)
colnames(table.rad.f1) <- c("F1","cost.constraints.violation")

ggplot(as.data.frame(table.rad.f1))+
  aes(x= cost.constraints.violation, y = F1, fill=cost.constraints.violation)+
  geom_bar(stat="identity")+
  xlab("Coût de la violation de la contrainte")+
  ylab("F1 mesure")+
  ggtitle("F1 mesure selon les cost constraints violation (pour un radial kernel)")


################################################# EVALUATION GAMMA

library(RColorBrewer)

#https://datatuts.com/svm-parameter-tuning/


#the Gaussian class boundaries dissipate as they get further from the support vectors.
#The gamma parameter determines how quickly this dissipation happens; larger values decrease the effect of any individual support vector.
#increasing gamma seem like a great idea, we could just reduce the effect of each of those clusters enough we could get nicely delineated boundaries:


mod.gam001 <- crossvalidate.SVM(kernel='radial',cost=100,gamma=0.01) #86.1% 
mod.gam002 <- crossvalidate.SVM(kernel='radial',cost=100,gamma=0.02) #86.5% 
mod.gam004 <- crossvalidate.SVM(kernel='radial',cost=100,gamma=0.04) #87.7% 
mod.gam01 <- crossvalidate.SVM(kernel='radial',cost=100,gamma=0.1) #86% 

vec.mod.gam<- as.data.frame(cbind(mod.gam001[1],mod.gam002[1],mod.gam004[1],mod.gam01[1]))
table.gam <- t(vec.mod.gam)
vec.mod.gam.f1 <- c("0.01","0.02","0.04","0.1")
# vec.mod.rad.f1 <- ordered(vec.mod.rad.f1, levels = c("0.01","0.02","0.04","0.1"))
table.gam<- cbind(table.gam,vec.mod.gam.f1)
colnames(table.gam) <- c("F1","table.gam")

#source : https://ggplot2.tidyverse.org/reference/scale_brewer.html

p <- ggplot(as.data.frame(table.gam))+
  scale_fill_continuous(low="blue", high="red")+
  aes(x= gamma, y = F1, fill=gamma)+
  geom_bar(stat="identity")+
  xlab("Gamma")+
  ylab("F1 mesure")+
  ggtitle("F1 mesure selon les gamma (pour un radial kernel de coût 100)")
# scale_fill_manual(values = c("red","blue","green","yellow"))
p + scale_fill_brewer() + theme_dark()


# ###################################################################
#                                                                   #
#       5.                                                          #
#          COMPARAISON DES MODELES PAR LA COURBE ROC                #
#                                                                   #     
# ###################################################################

############## Receiving Operator Characteristics (ROC)

#tracer la courbe ROC

#http://eric.univ-lyon2.fr/~ricco/cours/slides/roc_curve.pdf
#http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Spv_Learning_Curves.pdf

#fonction pour la courbe ROC

roc_curve <- function(y,score){
  #nombre d'exemples
  n <- length(y)
  #nombre d'exemples positifs
  pos <- sum(y)
  #nombre d'exemples négatifs
  neg <- n - pos
  #size of the target
  target <-seq(1,n,1)
  #valeurs de tri
  index <- sort(score,decreasing=T, index.return=T)
  sy <- y[index$ix]
  sscore <- score[index$ix]
  #somme cumulée des positifs
  c.pos <- cumsum(sy)
  #TVP - taux de vrais positifs
  tvp <- c.pos/pos
  tvp <- c(0,tvp)
  #somme cumulée des négatifs
  c.neg <- target - c.pos
  #TFP - taux de faux positifs
  tfp <- c.neg/neg
  tfp <- c(0,tfp)
  #resultat retourné
  return(list(x=tfp, y=tvp))
}

########################################## tracer la courbe SVM

#meilleur modele SVM
svm.mod<- svm(train.cr$NSP~., data= train.cr, probability = T, kernel="radial", cost=100, gamma=0.04, scale= TRUE) 
svm.pred <- predict(svm.mod,newdata = test.cr[,1:21], probability = T)

#recoder la variable NSP en binaire
y <- ifelse(test.cr$NSP=="p",1,0)
#recuperer la probabilité d'être pathologique "p"
svm.score <- attr(svm.pred,"probabilities")[,2]
#recuperer les coordonnées pour projeter dans le repere
svm.roc <- roc_curve(y,svm.score)
#tracer la courbe roc
plot(svm.roc$x,svm.roc$y,col="red",main="courbe ROC", xlab="TFP", ylab="TVP",type = "l")

################################## ajouter la courbe elasticnet

#meilleur elasticnet : alpha 0.1 et lambda 0.01
mod.elasticnet <- glmnet(x=data.matrix(train.cr[,1:21]), y=data.matrix(train.cr[,22]),alpha=0.1, family="binomial", standardize=FALSE)
print(summary(mod.elasticnet))
yenet2 <- predict(mod.elasticnet, data.matrix(test.cr[,1:21]),s=c(0.01),type="response")
print(summary(yenet2))
#roc
elasti.roc <- roc_curve(y,yenet2)
lines(elasti.roc$x,elasti.roc$y,type = "l", col="blue")

################################## ajouter la courbe neuralnet

#modele retenu pour le neuralnet

mod.final <- neuralnet(as.formula(formule),data=data.matrix(train.cr.bin),
                       err.fct = "sse", act.fct = "logistic", hidden=c(10), threshold=0.01,
                       linear.output=FALSE) #car on est en classement
plot(mod.final)
#test du modele sur le bloc numero k
pred.neural.final <- predict(mod.final, data.matrix(test.cr.bin[1:21]))
print(summary(pred.neural.final))
#traduire les modalites
NSP.bin.truth <- ifelse(test.cr.bin2[,"NSP.bin2"]>0.5,"p","n")
NSP.bin.pred <- ifelse(pred.final>0.5,"p","n")
#roc
neural.roc <- roc_curve(y,pred.neural.final)
lines(neural.roc$x,neural.roc$y,type = "l", col="yellow")

legend("topright",legend=c("elasticnet","SVM","neuralnet"),
       pch=19,col=c("red","blue","yellow"),cex=0.75,fill=1:2)