#regression logistique binomiale penalisee par une fonction de regularisation elasticnet 

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


