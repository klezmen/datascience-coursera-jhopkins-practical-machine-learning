# datascience-coursera-jhopkins-practical-machine-learning
The goal of this project is to predict how well a person is doing his or her exercise based on sensor data.
Predicting Barbell Movements from Sensor Data
========================================================

Wearable devices enable users to store large amounts of data collected through sensors. While most practical applications of sensor data are descriptive[1],  in this article we will go a step further and predict types of movements from gathered sensor data. 6 participants were asked to perform barbell lifts in 5 different ways, sensor data is stored and from this data a predictive model is created and assessed on accuracy. We will assess random forest and gbm (gradient boosing) on accurcay and computational efficiency. 

###Loading the Data
These are the required packages:

```{r}
library(caret)
library(doSNOW)
```
doSNOW is a package that allow parallelisation utilizing multiple cpu-cores[4].
To initialise it for 4 cores  whe do the following: 

```{r}
cl <- makeCluster(4)
registerDoSNOW(cl)


```

Let's load the data
```{r}

pml.training<- read.csv("./pml-training.csv")
pml.testing <- read.csv("./pml-testing.csv")
```
### 1. Inspection and Summary Descriptives 
Upon inspecting the features it became appearent that a lot of variables are empty 
```{r}
head(pml.training,5)
head(pml.testing,5)
```

```{r}
summary(pml.training)
```
A lot of variables seem to empty so let's inspect the missing data

```{r}
nacount<-sum(is.na(pml.training))
totalcell<-c(19622*158)
nacount<-(1287472)
nacount/totalcell
[1] 0.4152766
```

41% of all cells are empty. 
Let's remove the empty variables (like we have seen before with summary), and redundant variables (timestamp etc ) 




```{r}
variablesincluded <- c("roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt", 
                       "gyros_belt_x", "gyros_belt_y", "gyros_belt_z", "accel_belt_x", "accel_belt_y", 
                       "accel_belt_z", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", "roll_arm", 
                       "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_arm_x", "gyros_arm_y", 
                       "gyros_arm_z", "accel_arm_x", "accel_arm_y", "accel_arm_z", "magnet_arm_x", 
                       "magnet_arm_y", "magnet_arm_z", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", 
                       "total_accel_dumbbell", "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z", 
                       "accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z", "magnet_dumbbell_x", 
                       "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", "pitch_forearm", 
                       "yaw_forearm", "total_accel_forearm", "gyros_forearm_x", "gyros_forearm_y", 
                       "gyros_forearm_z", "accel_forearm_x", "accel_forearm_y", "accel_forearm_z", 
                       "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z")
```

Creating new dataframes for both the test - and trainingset without redundant features.
```{r}
pmltslct<-pml.training[,variablesincluded]
pmlttestslct<-pml.testing[,variablesincluded]
nacount<-sum(is.na(pmltslct))
nacount
[1] 0
```
Now the redundant features are removed, we have zero missing values.
###2. Modelfitting and Training

First we will try the random forest algorithm and inspect the acuracy. We expect this model to have some out of sample error, but at least it will provide a performance metric for model selection.

###2.1 Random forest


```{r}
modelFitrf <- train(pml.training$classe ~ ., method = "rf", allowParallel=TRUE,data = pmltslct)
summary(modelFitrf)
```
This model gives an Accuracy of .985 and an in sample error of (1-.985) .015 which is a very good performance.

```{r}
Bootstrapped (25 reps) Confusion Matrix 

(entries are percentages of table totals)
 
          Reference
Prediction    A    B    C    D    E
         A 28.5  0.1  0.0  0.0  0.0
         B  0.0 19.1  0.1  0.0  0.0
         C  0.0  0.0 17.2  0.3  0.0
         D  0.0  0.0  0.0 16.2  0.0
         E  0.0  0.0  0.0  0.0 18.3
```

Another interesting performance metric is the variable importance of the features included in the model. Roll_belt has by far the most imfluence on performance. There is no reason to drop any variable from the model as all of them have some influence on predictive performance.

```{r}
varImp(modelFitrf)
rf variable importance

  only 20 most important variables shown (out of 52)

                  Overall
roll_belt          100.00
yaw_belt            76.16
magnet_dumbbell_z   67.88
magnet_dumbbell_y   62.30
pitch_belt          60.77
pitch_forearm       58.82
magnet_dumbbell_x   51.11
roll_forearm        50.61
accel_belt_z        43.83
roll_dumbbell       42.69
accel_dumbbell_y    42.61
magnet_belt_y       41.24
magnet_belt_z       41.02
accel_dumbbell_z    37.10
roll_arm            36.00
accel_forearm_x     31.02
gyros_belt_z        30.12
accel_dumbbell_x    29.64
gyros_dumbbell_y    28.97
yaw_dumbbell        28.93
```




###2.2 Gradient Boosting with GBM
In the next model we will apply GBM. This algorithm is generally concidered to be more accurate than random forest because of its gradient descent algoritm minimising the error over multiple of iterations[2]. With GBM we expect a lower out of sample error than with a regular random forest model.

```{r}
modelFitgbm <- train(pml.training$classe ~ ., method = "gbm", allowParallel=TRUE,data = pmltslct)
summary(modelFitgbm) 
confusiontable<-confusionMatrix(modelFitgbm)

```
 
```{r}
9812 samples
 159 predictors
   5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Bootstrapped (25 reps) 

Summary of sample sizes: 201, 201, 201, 201, 201, 201, ... 

Resampling results across tuning parameters:

  mtry  Accuracy  Kappa    Accuracy SD  Kappa SD
  2     0.239     0.00353  0.0385       0.0148  
  117   0.74      0.668    0.0507       0.0634  
  6950  0.994     0.993    0.00774      0.00983 
```

This model performs slightly better than random forest (.994) although the computation time took a full day on a 4-core macbook pro with 16gb ram.
GBM is a lot more computationally expensive than random forest because of it’s gradient algorithm running over multiple iterations (finding minimum optima) so in the next model we try to find the optimum parameters with random forest, to see if we can outperform GBM.


###2.3 Random Forest with Cross Validation

In the next model we implement more cross validations so that the final model is an average of even more random forest  models . 


```{r}


modelFitrf2 <- train(pml.training$classe ~ ., method = "rf", allowParallel=TRUE,trControl=trainControl(method="cv",number=5),data =pmltslct)
```
Let's look at the results of the random forest model

```{r}
summary(modelFitrf2)

```
```{r}
Random Forest 

9812 samples
 159 predictors
   5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Out of Bag Resampling 

Summary of sample sizes:  

Resampling results across tuning parameters:

  mtry  Accuracy  Kappa
  2     0.264     0    
  117   0.816     0.765
  6950  1         1    


Accuracy was used to select the optimal model using  the largest value.
The final value used for the model was mtry = 6952. 
```

With cross validated random forest we managed to outperform the GBM with a 100% accuracy so this will be our final model. 

###3. Prediction
So now we are going to predict the classes of the test set with our final model

```{r}
predict(modelFitrf,newdata=pmlttestslct)
     B A B A A E D B A A B C B A E E A B B B
```
Which turn out to result in a 100% accuracy on the test set. An interseting observation is that in this case the out of sample cross validation error (0) is lower than the in sample error (.015). This is caused by the small size of the test set (20 observations) relative to the training set.



####Conclusion
Although normally we would expect the out of sample error to be higher than in the training set, we found in our results that the out of sample error was smaller (zero error). The accuracy of 1 generated by random forest with 5 fold cross validation was the result of having a large training set as opposed to a small test set.
GBM was more effective on the training set than a regular random forrest model but at the expensive of computational efficiency. Random forest with 5-fold cross validations where the final model is an average of multiple random forest models performs the most accurate with manageable computational load[5]. 


####Literature
1. Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science., pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6. 
2. Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
3. Friedman, J. H. "Greedy Function Approximation: A Gradient Boosting Machine." (Feb. 1999a)
4. Introduction to parallel computing in R - Michael J Koontz (April 2014)
5. Leo Breiman (2003). Manual for Setting Up, Using, and Understanding Random Forest V4.0. http://oz.berkeley.edu/users/breiman/Using_random_forests_v4.0.pdf
