#  Regression regularisee pour le classement
#Regression logistique multinomial
#source : http://eric.univ-lyon2.fr/~ricco/cours/slides/regularized_regression.pdf

#  Probleme : Cardiotocogrammes (CTG) des foetus a la maternite,
#detecter une souffrance foetale NSP = "p" (patho ou suspect)
#a partir des caracteristiques (p=21) du trace des CTG (n= 2126)

#install.packages("XLConnect")
library(dplyr)
library(xlsx)
library(XLConnect)
library(readr)

################################## 1. IMPORTATION BASE DE DONNEES, RECODAGE

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


############################### 2. ECHANTILLONS POUR L'APPRENTISSAGE, TEST

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


######################################## 3. CENTRER ET REDUIRE LES VA num

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
