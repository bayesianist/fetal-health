#orienter adroitement le paramètre de régularisation après avoir défini 
#un critère d’évaluation calculé sur un fichier test

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

#valeurs  d’initialisation  de    l’heuristique  d’optimisation. ?
