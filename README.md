# Classification Cardiac Arrhythmia
Classification of the Cardiac Arrhythmia using Spark ML in Scala and LSTM network in Java. This is the code for our paper titled "A Two-stage Big Data Analytics Framework with Real World Application using Spark ML and LSTM Networks" submitted to "Symmetry — Open Access Journal" (see http://www.mdpi.com/journal/symmetry). For the LSTM, DeepLearning4j has been used. 

# How to use the repository
## For Spark ML: 
The following classifiers are implemented to solve the classification problem:
- Logistic Regression
- Decision Trees
- Random Forest
- Multilayer Perceptron (MLP)

Nevertheless, we implemnted H2O = Sparkling Water version too. Take a look at ArrhythmiaPredictionH2O.scala class for this. 

Make sure that Spark is properly configured. Also, you need to have Maven installed on Linux. If you prefer, Eclipse/IntelliJ IDEA, make sure that Maven plugin and Scala plugins are installed.  

If everything is properly configured, you can create a uber jar containing all the dependencies and execute the jar. Alternatively, you can execute ech implementation as a stand-alone Scala project from your favourite IDE. 


## For DeepLearning4j: 
The Long Short-term Memory (LSTM) network has been implemented to solve the classification problem. The following are prerequisites when working with DL4J:
- Java 1.8+ (64-bit only)
- Apache Maven for automated build and dependency manager
- IntelliJ IDEA or Eclipse IDE.

For more information on how to configure DeepLearning4j, please refer to https://deeplearning4j.org/. If everything is properly configured, you can create a uber jar containing all the dependencies and execute the jar. Alternatively, you can execute ech implementation as a stand-alone Java project from your favourite IDE. 
