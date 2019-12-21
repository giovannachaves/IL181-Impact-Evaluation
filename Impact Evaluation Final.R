# Installing and loading required packages and data #

install.packages("PNADcIBGE")
if (!require("devtools")) install.packages("devtools")
devtools::install_github("JasjeetSekhon/Matching")
if (!require("devtools")) install.packages("devtools")
devtools::install_github("JasjeetSekhon/rgenoud")
install.packages("rbounds")
install.packages("tidyverse")
install.packages("caret")
install.packages("leaps")
install.packages("glmnet")

library(PNADcIBGE)
library(Matching)
library(rgenoud)
library(rbounds)
library(plyr)
library(tidyverse)
library(caret)
library(leaps)
library(MASS)
library(glmnet)
library(ggplot2)

setwd("~/Desktop/Capstone Data/PNAD Contínua")

dados2016 <- read_pnadc("PNADC_022016_educacao.txt", "Input_PNADC_trimestral_educacao_20191016.txt")
dados2017 <- read_pnadc("PNADC_022017_educacao.txt", "Input_PNADC_trimestral_educacao_20191016.txt")
dados2018 <- read_pnadc("PNADC_022018_educacao.txt", "Input_PNADC_trimestral_educacao_20191016.txt")

dadostotal <- rbind(dados2016, dados2017, dados2018)

# Defining treatment # 
dadostotal <- subset(dadostotal, V3021!=(is.na(dadostotal$V3021)))
dadostotal$Tr <- rep(0, length(dadostotal$V3021))
dadostotal$Tr[which(dadostotal$V3021 == 1)] <- 1

# Descriptive statistics #

withoutignored <- subset(dadostotal, V2010!=9)
withoutNA <- subset(dadostotal, V3018!=0)
ggplot(data = dadostotal, aes(dadostotal$Tr)) + geom_histogram(aes(y = stat(count / sum(count))), col = "black", bins = 3, fill = "olivedrab4", alpha = 0.7) + labs(title = "Distribution of individuals by treatment condition", y = "Density",x = "Condition: Control (0), Treatment (1)")
ggplot(data = dadostotal, aes(dadostotal$UF)) + geom_histogram(col = "black", fill = "deepskyblue4", bins = 53, alpha = 0.7) + labs(title = "Distribution of individuals by state", subtitle = "for control (0) and treatment (1)", y = "Count of individuals",x = "State code") + facet_wrap(~Tr)
ggplot(data = dadostotal, aes(dadostotal$V1022)) + geom_histogram(aes(y = stat(count / sum(count))), col = "black", fill = "deepskyblue2", bins = 2, alpha = 0.7) + labs(title = "Distribution of individuals by household area", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Household area, urban (1) and rural (2)") + facet_wrap(~Tr)
ggplot(data = dadostotal, aes(dadostotal$V1023)) + geom_histogram(aes(y = stat(count / sum(count))), bins = 4, col = "black", fill = "chartreuse4", alpha = 0.9) + labs(title = "Distribution of individuals by household location", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Household location within state") + facet_wrap(~Tr)
ggplot(data = dadostotal, aes(dadostotal$V2007)) + geom_histogram(aes(y = stat(count / sum(count))), bins = 2, col = "black", fill = "indianred3", alpha = 0.9) + labs(title = "Distribution of individuals by gender", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Gender, male (1) and female (2)") + facet_wrap(~Tr)
ggplot(data = dadostotal, aes(dadostotal$V2009)) + geom_histogram(aes(y = stat(count / sum(count))), col = "black", fill = "paleturquoise2", alpha = 0.9) + labs(title = "Distribution of individuals by age", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Age") + facet_wrap(~Tr)
ggplot(data = withoutignored, aes(withoutignored$V2010)) + geom_histogram(aes(y = stat(count / sum(count))), bins = 5, col = "black", fill = "turquoise4", alpha = 0.9) + labs(title = "Distribution of individuals by race", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Race") + facet_wrap(~Tr)
ggplot(data = withoutNA, aes(withoutNA$V3018)) + geom_histogram(aes(y = stat(count / sum(count))), bins = 3, col = "black", fill = "darkorange2", alpha = 0.9) + labs(title = "Distribution of individuals by high school type", subtitle = "for control (0) and treatment (1)", y = "Density", x = "High school type: private (1), public (2), mix (3)") + facet_wrap(~Tr)

# Do LASSO regression to select variables #
# Code retrieved from: http://www.sthda.com/english/articles/36-classification-methods-essentials/149-penalized-logistic-regression-essentials-in-r-ridge-lasso-and-elastic-net/#loading-required-r-packages

training.samples <- dadostotal$Tr %>% createDataPartition(p = 0.8, list = FALSE) 
# Divide data between train and test
train.data  <- dadostotal[training.samples, ]
test.data <- dadostotal[-training.samples, ]
x_lasso <- cbind(train.data$UF, train.data$V1022, train.data$V1023, train.data$V2007, train.data$V2009, train.data$V2010, train.data$V3018)
y_lasso <- train.data$Tr
cv.lasso <- cv.glmnet(x_lasso, y_lasso, alpha = 1, family = "binomial") 
# Finding the optimal value of lambda that minimizes the cross-validation error
plot(cv.lasso)
cv.lasso$lambda.min 
# Value of lanbda that minimizes the prediction error is 0.0006712791
model_lasso <- glmnet(x_lasso, y_lasso, alpha = 1, family = "binomial", lambda = cv.lasso$lambda.min)
coef(model_lasso)
coef(cv.lasso, cv.lasso$lambda.min) 
# None of our variables were reduced to 0, which is to say they all increase accuracy.

# Estimate the propensity score model #

dadostotal$Ano <- as.numeric(dadostotal$Ano)
dadostotal$UF <- as.numeric(dadostotal$UF)
dadostotal$V1022 <- as.numeric(dadostotal$V1022)
dadostotal$V1023 <- as.numeric(dadostotal$V1023)
dadostotal$V2007 <- as.numeric(dadostotal$V2007)
dadostotal$V2009 <- as.numeric(dadostotal$V2009)
dadostotal$V2010 <- as.numeric(dadostotal$V2010)
dadostotal$V3018 <- as.numeric(dadostotal$V3018)
dadostotal$V3018[is.na(dadostotal$V3018)] <- 0 # replacing NAs with 0 so still able to run

glm1  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadostotal)
summary(glm1)

# Setting covariates and outcome variables #

X  <- cbind(glm1$fitted, dadostotal$Ano, dadostotal$UF, dadostotal$V1022, dadostotal$V1023, dadostotal$V2007, dadostotal$V2009, dadostotal$V2010, dadostotal$V3018)
Y1 <- as.numeric(dadostotal$V4001)

# Creating a subset of 1% of the data from which to choose the weights from given that my current dataset has 509156. #

set.seed(181) 
mask <- sample(1:nrow(dadostotal), 5091, replace = FALSE)
dados_sample <- dadostotal[mask, ]
pscore_sample <- glm1$fitted.values[mask]
X_sample  <- cbind(pscore_sample, dados_sample$Ano, dados_sample$UF, dados_sample$V1022, dados_sample$V1023, dados_sample$V2007, dados_sample$V2009, dados_sample$V2010, dados_sample$V3018)

# # Applying genetic matching # # 

genout <- GenMatch(Tr=dados_sample$Tr, X=X_sample, BalanceMatrix = X_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)

# Using weights from the sample data to match at the population level #

mout1 <- Match(Y=Y1, Tr=dadostotal$Tr, X=X, estimand="ATT", M=1, Weight.matrix=genout, version = 'fast', ties = FALSE)
summary(mout1)

# Let's determine if balance has actually been obtained on the variables of interest #

mb <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadostotal, match.out=mout1, nboots=1000, paired = FALSE)

# Sensitivity
hlsens(mout1, pr = 0.1, Gamma = 3, GammaInc = 0.1)

# # Repeat for all the other outcome variables # #

# # Employment
dadosY2 <- subset(dadostotal, V4003!=(is.na(dadostotal$V4003)))
Y2 <- as.numeric(dadosY2$V4003)
glm2  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY2)
mask2 <- sample(1:nrow(dadosY2), 2020, replace = FALSE)
dadosY2_sample <- dadosY2[mask2, ]
pscore2 <- glm2$fitted.values[mask2]
X2_sample  <- cbind(pscore2, dadosY2_sample$Ano, dadosY2_sample$UF, dadosY2_sample$V1022, dadosY2_sample$V1023, dadosY2_sample$V2007, dadosY2_sample$V2009, dadosY2_sample$V2010, dadosY2_sample$V3018)
genout2 <- GenMatch(Tr=dadosY2_sample$Tr, X=X2_sample, BalanceMatrix = X2_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X2 <- cbind(glm2$fitted.values,dadosY2$Ano, dadosY2$UF, dadosY2$V1022, dadosY2$V1023, dadosY2$V2007, dadosY2$V2009, dadosY2$V2010, dadosY2$V3018)
mout2 <- Match(Y=Y2, Tr=dadosY2$Tr, X=X2, estimand="ATT", M=1, Weight.matrix=genout2, version = 'fast')
summary(mout2)
mb2 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY2, match.out=mout2, nboots=2000, paired = FALSE)
hlsens(mout2, pr = 0.1, Gamma = 3, GammaInc = 0.1)

# # Formality
dadosY3 <- subset(dadostotal, V4029!=(is.na(dadostotal$V4029)))
Y3 <- as.numeric(dadosY3$V4029)
glm3  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY3)
mask3 <- sample(1:nrow(dadosY3), 2008, replace = FALSE)
dadosY3_sample <- dadosY3[mask3, ]
pscore3 <- glm3$fitted.values[mask3]
X3_sample  <- cbind(pscore3, dadosY3_sample$Ano, dadosY3_sample$UF, dadosY3_sample$V1022, dadosY3_sample$V1023, dadosY3_sample$V2007, dadosY3_sample$V2009, dadosY3_sample$V2010, dadosY3_sample$V3018)
genout3 <- GenMatch(Tr=dadosY3_sample$Tr, X=X3_sample, BalanceMatrix = X3_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X3 <- cbind(glm3$fitted.values,dadosY3$Ano, dadosY3$UF, dadosY3$V1022, dadosY3$V1023, dadosY3$V2007, dadosY3$V2009, dadosY3$V2010, dadosY3$V3018)
mout3 <- Match(Y=Y3, Tr=dadosY3$Tr, X=X3, estimand="ATT", M=1, Weight.matrix=genout3, version = 'fast')
summary(mout3)
mb3 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY3, match.out=mout3, nboots=2000, paired = FALSE)
hlsens(mout3, pr = 0.1, Gamma = 3, GammaInc = 0.1)

# # Monthly earnings 
Y4 <- as.numeric(dadostotal$V403312[which(dadostotal$V4033==1 & dadostotal$V40331==1)])
dadosY4 <- subset(dadostotal, V4033==1 & dadostotal$V40331==1)
glm4  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY4)
mask4 <- sample(1:nrow(dadosY4), 3197, replace = FALSE)
dadosY4_sample <- dadosY4[mask4, ]
pscore4 <- glm4$fitted.values[mask4]
X4_sample  <- cbind(pscore4, dadosY4_sample$Ano, dadosY4_sample$UF, dadosY4_sample$V1022, dadosY4_sample$V1023, dadosY4_sample$V2007, dadosY4_sample$V2009, dadosY4_sample$V2010, dadosY4_sample$V3018)
genout4 <- GenMatch(Tr=dadosY4_sample$Tr, X=X4_sample, BalanceMatrix = X4_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X4  <- cbind(glm4$fitted.values,dadosY4$Ano, dadosY4$UF, dadosY4$V1022, dadosY4$V1023, dadosY4$V2007, dadosY4$V2009, dadosY4$V2010, dadosY4$V3018)
mout4 <- Match(Y=Y4, Tr=dadosY4$Tr, X=X4, estimand="ATT", M=1, Weight.matrix=genout4, version = 'fast')
summary(mout4)
mb4 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY4, match.out=mout4, nboots=2000, paired = FALSE)
hlsens(mout4, pr = 0.1, Gamma = 3, GammaInc = 0.1)

# # Hours worked 
dadosY5 <- subset(dadostotal, V4039!=(is.na(dadostotal$V4039)))
Y5 <- as.numeric(dadosY5$V4039)
glm5  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY5)
mask5 <- sample(1:nrow(dadosY5), 3282, replace = FALSE)
dadosY5_sample <- dadosY5[mask5, ]
pscore5 <- glm5$fitted.values[mask5]
X5_sample  <- cbind(pscore5, dadosY5_sample$Ano, dadosY5_sample$UF, dadosY5_sample$V1022, dadosY5_sample$V1023, dadosY5_sample$V2007, dadosY5_sample$V2009, dadosY5_sample$V2010, dadosY5_sample$V3018)
genout5 <- GenMatch(Tr=dadosY5_sample$Tr, X=X5_sample, BalanceMatrix = X5_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X5  <- cbind(glm5$fitted.values,dadosY5$Ano, dadosY5$UF, dadosY5$V1022, dadosY5$V1023, dadosY5$V2007, dadosY5$V2009, dadosY5$V2010, dadosY5$V3018)
mout5 <- Match(Y=Y5, Tr=dadosY5$Tr, X=X5, estimand="ATT", M=1, Weight.matrix=genout5, version = 'fast')
summary(mout5)
mb5 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY5, match.out=mout5, nboots=1000, paired = FALSE)
hlsens(mout5, pr = 0.1, Gamma = 3, GammaInc = 0.1)

# # Recebe Bolsa Família, using on 10% of the data
dadosY6 <- subset(dadostotal, VI5001A2!=(is.na(dadostotal$VI5001A2)))
Y6 <- as.numeric(dadosY6$VI5001A2)
glm6  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY6)
mask6 <- sample(1:nrow(dadosY6), 235, replace = FALSE)
dadosY6_sample <- dadosY6[mask6, ]
pscore6 <- glm6$fitted.values[mask6]
X6_sample  <- cbind(pscore6, dadosY6_sample$Ano, dadosY6_sample$UF, dadosY6_sample$V1022, dadosY6_sample$V1023, dadosY6_sample$V2007, dadosY6_sample$V2009, dadosY6_sample$V2010, dadosY6_sample$V3018)
genout6 <- GenMatch(Tr=dadosY6_sample$Tr, X=X6_sample, BalanceMatrix = X6_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X6  <- cbind(glm6$fitted.values,dadosY6$Ano, dadosY6$UF, dadosY6$V1022, dadosY6$V1023, dadosY6$V2007, dadosY6$V2009, dadosY6$V2010, dadosY6$V3018)
mout6 <- Match(Y=Y6, Tr=dadosY6$Tr, X=X6, estimand="ATT", M=1, Weight.matrix=genout6, version = 'fast')
summary(mout6)
mb6 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY6, match.out=mout6, nboots=2000, paired = FALSE)
hlsens(mout6, pr = 0.1, Gamma = 3, GammaInc = 0.1)

# # Years of study
Y7 <- as.numeric(dadostotal$VD3005)
mout7 <- Match(Y=Y7, Tr=dadostotal$Tr, X=X, estimand="ATT", M=1, Weight.matrix=genout, version = 'fast', ties = FALSE)
summary(mout7)
mb7 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadostotal, match.out=mout7, nboots=500, paired = FALSE)
hlsens(mout7, pr = 0.1, Gamma = 3, GammaInc = 0.1)

# # # But what if we define treatment differently?
dadostotal <- subset(dadostotal, V3021!=(is.na(dadostotal$V3021)))
dadostotal$Tr <- rep(0, length(dadostotal$V3021))
dadostotal$Tr[which(dadostotal$V3022 == 2 | dadostotal$V3022 == 3)] <- 1
dadostotal$Tr <- as.numeric(dadostotal$Tr)

# Run everything again but for a different treatment definition!

dadostotal$Ano <- as.numeric(dadostotal$Ano)
dadostotal$UF <- as.numeric(dadostotal$UF)
dadostotal$V1022 <- as.numeric(dadostotal$V1022)
dadostotal$V1023 <- as.numeric(dadostotal$V1023)
dadostotal$V2007 <- as.numeric(dadostotal$V2007)
dadostotal$V2009 <- as.numeric(dadostotal$V2009)
dadostotal$V2010 <- as.numeric(dadostotal$V2010)
dadostotal$V3018 <- as.numeric(dadostotal$V3018)
dadostotal$V3018[is.na(dadostotal$V3018)] <- 0 # replacing NAs with 0 so still able to run

withoutignored <- subset(dadostotal, V2010!=9)
withoutNA <- subset(dadostotal, V3018!=0)
ggplot(data = dadostotal, aes(dadostotal$Tr)) + geom_histogram(aes(y = stat(count / sum(count))), col = "black", bins = 3, fill = "olivedrab4", alpha = 0.7) + labs(title = "Distribution of individuals by treatment condition", y = "Density",x = "Condition: Control (0), Treatment (1)")
table(dadostotal$Tr, dadostotal$Ano)
ggplot(data = dadostotal, aes(dadostotal$UF)) + geom_histogram(col = "black", fill = "deepskyblue4", bins = 53, alpha = 0.7) + labs(title = "Distribution of individuals by state", subtitle = "for control (0) and treatment (1)", y = "Count of individuals",x = "State code") + facet_wrap(~Tr)
ggplot(data = dadostotal, aes(dadostotal$V1022)) + geom_histogram(aes(y = stat(count / sum(count))), col = "black", fill = "deepskyblue2", bins = 2, alpha = 0.7) + labs(title = "Distribution of individuals by household area", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Household area, urban (1) and rural (2)") + facet_wrap(~Tr)
ggplot(data = dadostotal, aes(dadostotal$V1023)) + geom_histogram(aes(y = stat(count / sum(count))), bins = 4, col = "black", fill = "chartreuse4", alpha = 0.9) + labs(title = "Distribution of individuals by household location", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Household location within state") + facet_wrap(~Tr)
ggplot(data = dadostotal, aes(dadostotal$V2007)) + geom_histogram(aes(y = stat(count / sum(count))), bins = 2, col = "black", fill = "indianred3", alpha = 0.9) + labs(title = "Distribution of individuals by gender", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Gender, male (1) and female (2)") + facet_wrap(~Tr)
ggplot(data = dadostotal, aes(dadostotal$V2009)) + geom_histogram(aes(y = stat(count / sum(count))), col = "black", fill = "paleturquoise2", alpha = 0.9) + labs(title = "Distribution of individuals by age", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Age") + facet_wrap(~Tr)
ggplot(data = withoutignored, aes(withoutignored$V2010)) + geom_histogram(aes(y = stat(count / sum(count))), bins = 5, col = "black", fill = "turquoise4", alpha = 0.9) + labs(title = "Distribution of individuals by race", subtitle = "for control (0) and treatment (1)", y = "Density", x = "Race") + facet_wrap(~Tr)
ggplot(data = withoutNA, aes(withoutNA$V3018)) + geom_histogram(aes(y = stat(count / sum(count))), bins = 4, col = "black", fill = "darkorange2", alpha = 0.9) + labs(title = "Distribution of individuals by high school type", subtitle = "for control (0) and treatment (1)", y = "Density", x = "High school type: private (1), public (2), mix (3)") + facet_wrap(~Tr)

glm1  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadostotal)
summary(glm1)
X  <- cbind(glm1$fitted, dadostotal$Ano, dadostotal$UF, dadostotal$V1022, dadostotal$V1023, dadostotal$V2007, dadostotal$V2009, dadostotal$V2010, dadostotal$V3018)
Y1 <- as.numeric(dadostotal$V4001)

set.seed(1921) 
mask <- sample(1:nrow(dadostotal), 5091, replace = FALSE)
dados_sample <- dadostotal[mask, ]
pscore_sample <- glm1$fitted.values[mask]
X_sample  <- cbind(pscore_sample, dados_sample$Ano, dados_sample$UF, dados_sample$V1022, dados_sample$V1023, dados_sample$V2007, dados_sample$V2009, dados_sample$V2010, dados_sample$V3018)
genout <- GenMatch(Tr=dados_sample$Tr, X=X_sample, BalanceMatrix = X_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
mout1 <- Match(Y=Y1, Tr=dadostotal$Tr, X=X, estimand="ATT", M=1, Weight.matrix=genout)
summary(mout1)
mb <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadostotal, match.out=mout1, nboots=2000, paired = FALSE)
hlsens(mout1, pr = 0.1, Gamma = 3, GammaInc = 0.1)

Y7 <- as.numeric(dadostotal$VD3005)
mout7 <- Match(Y=Y7, Tr=dadostotal$Tr, X=X, estimand="ATT", M=1, Weight.matrix=genout, version = 'fast', ties = FALSE)
summary(mout7)
mb7 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadostotal, match.out=mout7, nboots=500, paired = FALSE)
hlsens(mout7, pr = 0.1, Gamma = 3, GammaInc = 0.1)

dadosY2 <- subset(dadostotal, V4003!=(is.na(dadostotal$V4003)))
Y2 <- as.numeric(dadosY2$V4003)
glm2  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY2)
mask2 <- sample(1:nrow(dadosY2), 2020, replace = FALSE)
dadosY2_sample <- dadosY2[mask2, ]
pscore2 <- glm2$fitted.values[mask2]
X2_sample  <- cbind(pscore2, dadosY2_sample$Ano, dadosY2_sample$UF, dadosY2_sample$V1022, dadosY2_sample$V1023, dadosY2_sample$V2007, dadosY2_sample$V2009, dadosY2_sample$V2010, dadosY2_sample$V3018)
genout2 <- GenMatch(Tr=dadosY2_sample$Tr, X=X2_sample, BalanceMatrix = X2_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X2 <- cbind(glm2$fitted.values,dadosY2$Ano, dadosY2$UF, dadosY2$V1022, dadosY2$V1023, dadosY2$V2007, dadosY2$V2009, dadosY2$V2010, dadosY2$V3018)
mout2 <- Match(Y=Y2, Tr=dadosY2$Tr, X=X2, estimand="ATT", M=1, Weight.matrix=genout2, version = 'fast')
summary(mout2)
mb2 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY2, match.out=mout2, nboots=2000, paired = FALSE)
hlsens(mout2, pr = 0.1, Gamma = 3, GammaInc = 0.1)

dadosY3 <- subset(dadostotal, V4029!=(is.na(dadostotal$V4029)))
Y3 <- as.numeric(dadosY3$V4029)
glm3  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY3)
mask3 <- sample(1:nrow(dadosY3), 2008, replace = FALSE)
dadosY3_sample <- dadosY3[mask3, ]
pscore3 <- glm3$fitted.values[mask3]
X3_sample  <- cbind(pscore3, dadosY3_sample$Ano, dadosY3_sample$UF, dadosY3_sample$V1022, dadosY3_sample$V1023, dadosY3_sample$V2007, dadosY3_sample$V2009, dadosY3_sample$V2010, dadosY3_sample$V3018)
genout3 <- GenMatch(Tr=dadosY3_sample$Tr, X=X3_sample, BalanceMatrix = X3_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X3 <- cbind(glm3$fitted.values,dadosY3$Ano, dadosY3$UF, dadosY3$V1022, dadosY3$V1023, dadosY3$V2007, dadosY3$V2009, dadosY3$V2010, dadosY3$V3018)
mout3 <- Match(Y=Y3, Tr=dadosY3$Tr, X=X3, estimand="ATT", M=1, Weight.matrix=genout3, version = 'fast')
summary(mout3)
mb3 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY3, match.out=mout3, nboots=2000, paired = FALSE)
hlsens(mout3, pr = 0.1, Gamma = 3, GammaInc = 0.1)

Y4 <- as.numeric(dadostotal$V403312[which(dadostotal$V4033==1 & dadostotal$V40331==1)])
dadosY4 <- subset(dadostotal, V4033==1 & dadostotal$V40331==1)
glm4  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY4)
mask4 <- sample(1:nrow(dadosY4), 3197, replace = FALSE)
dadosY4_sample <- dadosY4[mask4, ]
pscore4 <- glm4$fitted.values[mask4]
X4_sample  <- cbind(pscore4, dadosY4_sample$Ano, dadosY4_sample$UF, dadosY4_sample$V1022, dadosY4_sample$V1023, dadosY4_sample$V2007, dadosY4_sample$V2009, dadosY4_sample$V2010, dadosY4_sample$V3018)
genout4 <- GenMatch(Tr=dadosY4_sample$Tr, X=X4_sample, BalanceMatrix = X4_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X4  <- cbind(glm4$fitted.values,dadosY4$Ano, dadosY4$UF, dadosY4$V1022, dadosY4$V1023, dadosY4$V2007, dadosY4$V2009, dadosY4$V2010, dadosY4$V3018)
mout4 <- Match(Y=Y4, Tr=dadosY4$Tr, X=X4, estimand="ATT", M=1, Weight.matrix=genout4, version = 'fast')
summary(mout4)
mb4 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY4, match.out=mout4, nboots=2000, paired = FALSE)
hlsens(mout4, pr = 0.1, Gamma = 3, GammaInc = 0.1)

dadosY5 <- subset(dadostotal, V4039!=(is.na(dadostotal$V4039)))
Y5 <- as.numeric(dadosY5$V4039)
glm5  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY5)
mask5 <- sample(1:nrow(dadosY5), 3282, replace = FALSE)
dadosY5_sample <- dadosY5[mask5, ]
pscore5 <- glm5$fitted.values[mask5]
X5_sample  <- cbind(pscore5, dadosY5_sample$Ano, dadosY5_sample$UF, dadosY5_sample$V1022, dadosY5_sample$V1023, dadosY5_sample$V2007, dadosY5_sample$V2009, dadosY5_sample$V2010, dadosY5_sample$V3018)
genout5 <- GenMatch(Tr=dadosY5_sample$Tr, X=X5_sample, BalanceMatrix = X5_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X5  <- cbind(glm5$fitted.values,dadosY5$Ano, dadosY5$UF, dadosY5$V1022, dadosY5$V1023, dadosY5$V2007, dadosY5$V2009, dadosY5$V2010, dadosY5$V3018)
mout5 <- Match(Y=Y5, Tr=dadosY5$Tr, X=X5, estimand="ATT", M=1, Weight.matrix=genout5, version = 'fast')
summary(mout5)
mb5 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY5, match.out=mout5, nboots=1000, paired = FALSE)
hlsens(mout5, pr = 0.1, Gamma = 3, GammaInc = 0.1)

dadosY6 <- subset(dadostotal, VI5001A2!=(is.na(dadostotal$VI5001A2)))
Y6 <- as.numeric(dadosY6$VI5001A2)
glm6  <- glm(Tr ~ UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, family=binomial, data=dadosY6)
mask6 <- sample(1:nrow(dadosY6), 235, replace = FALSE)
dadosY6_sample <- dadosY6[mask6, ]
pscore6 <- glm6$fitted.values[mask6]
X6_sample  <- cbind(pscore6, dadosY6_sample$Ano, dadosY6_sample$UF, dadosY6_sample$V1022, dadosY6_sample$V1023, dadosY6_sample$V2007, dadosY6_sample$V2009, dadosY6_sample$V2010, dadosY6_sample$V3018)
genout6 <- GenMatch(Tr=dadosY6_sample$Tr, X=X6_sample, BalanceMatrix = X6_sample, estimand="ATT", M=1, pop.size=200, max.generations=1000, wait.generations=20)
X6  <- cbind(glm6$fitted.values,dadosY6$Ano, dadosY6$UF, dadosY6$V1022, dadosY6$V1023, dadosY6$V2007, dadosY6$V2009, dadosY6$V2010, dadosY6$V3018)
mout6 <- Match(Y=Y6, Tr=dadosY6$Tr, X=X6, estimand="ATT", M=1, Weight.matrix=genout6, version = 'fast')
summary(mout6)
mb6 <- MatchBalance(Tr ~ Ano + UF + V1022 + V1023 + V2007 + V2009 + V2010 + V3018, data=dadosY6, match.out=mout6, nboots=2000, paired = FALSE)
hlsens(mout6, pr = 0.1, Gamma = 3, GammaInc = 0.1)
