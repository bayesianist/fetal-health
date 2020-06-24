#comparer les performances de deux algorithmes d’apprentissage supervisé
#à l’aide de l’erreur calculée en validation croisée.

# source : https://www.r-bloggers.com/fitting-a-neural-network-in-r-neuralnet-package/

#estimation de la MSE et son IC
boxplot(cv.error,xlab='MSE CV',col='cyan',
        border='blue',names='CV error (MSE)',
        main='CV error (MSE) for NN',horizontal=TRUE)

############## Receiving Operator Characteristics

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

