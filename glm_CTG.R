#classement
#Regression logistique multinomial regularisee
#Modele penalise par une fonction de regularisation elasticnet 

#sources : http://eric.univ-lyon2.fr/~ricco/cours/slides/regularized_regression.pdf
#http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Ridge_Elasticnet_R.pdf

#techniques de re-echantillonnage par validation croisee
#destinées a estimer l’erreur de prediction à partir de la totalite des donnees dispo

#source : http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_Validation_Croisee_Suite.pdf 

#Probleme : Cardiotocogrammes du portugal
#detecter une souffrance foetale (response a K =3 niveaux, normal suspet patho)
#a partir de la description du CTG  
#avec n = 2126 et p=21

install.packages("rpart")
library(rpart)
library(glmnet)

######################################## Modelisation sur 10-CV

# CV pour evaluer les perf ac le rate error en CV : moy des rate error de chaque fold
# un estimateur de meilleure qualité que le taux d’erreur en resubstitution.

##################### 1. determiner le num de bloc de chaque obs

n <- nrow(ctg.cr) #nb d'obs
K <- 10 # pour 10-CV, par defaut: 5 x 2-CV préconise par  DIETTERICH
taille <- n%/%K #donne la taille des K blocs
set.seed(5) #pour obtenir la meme seq tout le tmps
alea <- runif(n) #random sampling : indices aleatoires pour les obs
rang <- rank(alea) #asso son indice à chaq obs
bloc <- (rang-1) %/% taille+1 #associe chaq obs à un bloc
bloc <- as.factor(bloc)
print(summary(bloc)) #10 blocs de 212 + 1 bloc de 6






####################### 2. test1 : regression logistique non regularise

all.err1 <- numeric(0) #lancer la CV
for (k in 1:K) {
  #data de train sur tous les individus sauf le bloc k
  train <- data.matrix(ctg.cr[bloc !=k,])
  #apprendre le mod (std: defini les descripteurs sur la meme echelle)
  cv.glmnet1 <- glmnet(x = train[,-22] , y = train[,22], family ="multinomial", 
                standardize=TRUE, lambda=0) 
  print(cv.glmnet1)
  #sur le bloc k de test
  test <- data.matrix(ctg.cr[bloc==k,])
  #appliquer le modele (class : produit la modalite de la class corres a la proba max, s=penalite)
  pred <- predict(cv.glmnet1, test[,-22], type = "class", s=c(0))
  print(pred)
  #matrice de confusion
  mc <- table(test[,22],pred)
  #taux d'erreur
  er1 <- sum(test[,22] !=pred)/nrow(test)
  #conserver
  all.err1 <- rbind(all.err1, er1)
}

#afficher les coefficients
#print(glmnet1$beta) #aucun à 0

#vecteur des erreurs recueillies
print(all.err1)
#erreur en CV, moyenne pondérée car blocs de taille presque identiques
err.cv1 <- mean(all.err1)
print(err.cv1) #0.106 deja pas mal...








install.packages("e1071")
library(caret)
library(e1071)

######################### 3. Mod 2 : Regression Ridge (lambda>0 et alpha=0)

# methode 2 : cv direct

# split 70 train | 30 test, validation set approach
set.seed(2) #fixe la graine du tirage aleatoire
training.idx <- createDataPartition(ctg.cr$NSP, p=0.7, list=FALSE) #stratifier sur Y (NSP)
#creer le jeu de train
train <- ctg.cr[training.idx,]
dim(train) #1490 obs
round(table(train$NSP)/nrow(train),2) #78% sont sains, 14% suspects, 8% patho
#creer le jeu de test
test <- ctg.cr[-training.idx,]
dim(test) #636
round(table(test$NSP)/nrow(test),2) #meme repartition
#attention glmnet prend en argument que des data sous forme matrix
Xtrain = data.matrix(train[,-22]) 
ytrain = data.matrix(train[,22])
Xtest = data.matrix(test[,-22]) 
ytest = data.matrix(test[,22])

#fit CV (keep en memoire les gpe des CV pour pv reproduire)
cv.ridge2 <- cv.glmnet(x = Xmatrix, y = ymatrix, family="multinomial", type.measure="class", nfolds=10, alpha=0, keep=TRUE)
#graph missclass error dans l'espace des log(lambda)
plot(cv.ridge2) #lignes en points au niveau des lambda: -2.7 (-3.9) pour min le missclass err avec parcimonie (ou non).
#chercher a optiminer le lambda : afficher la liste | missclass err | nb de coeff !0
print(cbind(cv.ridge2$lambda, cv.ridge2$cvm, cv.ridge2$nzero)) #toujours a 21 car ridge donc pas de selection
#min de l'erreur en CV
print(min(cv.ridge2$cvm)) #0.109 pas mieux....
#puis chercher le log du lambda correspondant a l'erreur min (regle de l'ecart type)
print(log(cv.ridge2$lambda.1se)) #log(lambda): -2.68
#inspection des coefficients Beta de la regression pour lambda min
print(coef(cv.ridge2, s="lambda.min")) #max des abs -> VA le + influent / la reg (DS: patho, DP : sain)

#prediction sur l'echantillon test
cv.pred2 <- predict(cv.ridge2,Xtest, s=c(cv.ridge2$lambda.min, cv.ridge2$lambda.1se),
                    type="class")
print(head(cv.pred2, n=20)) #col1 :lambda* (2: lambda**)
#missclass rate : lambda.min *
print(sum(test$NSP != cv.pred2[,1])/nrow(test)) #0.100
#missclass rate : lambda.1se **
print(sum(test$NSP != cv.pred2[,2])/nrow(test)) #0.108


