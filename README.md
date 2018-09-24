# How to use this repository? 
Classification of the Cardiac Arrhythmia and indentifying suspicious URLs for online learning using Spark ML in Scala and LSTM network in Scala and Java. However, for the LSTM, DeepLearning4j has been used. 

This is the code for our paper titled "A Two-stage Big Data Analytics Framework with Real World Applications using Spark ML and LSTM Networks". This papers has been submitted to "Symmetry — Open Access Journal" (see http://www.mdpi.com/journal/symmetry). 

## Spark MLlib-based classifers: 
The following classifiers have been implemented to solve both the classification problems in a 2 stage cascading style:
- Logistic Regression
- Decision Trees
- Random Forest
- Multilayer Perceptron (MLP)

Nevertheless, we implemnted Spark + H2O (aka. Sparkling Water) versions too. Take a look at the ArrhythmiaPredictionH2O.scala and URLReputationH2O.scala classes for the classification of the Cardiac Arrhythmia and indentifying suspicious URLs respectively. 

Make sure that Spark is properly configured. Also, you need to have Maven installed on Linux. If you prefer, Eclipse/IntelliJ IDEA, make sure that Maven plugin and Scala plugins are installed.  

If everything is properly configured, you can create a uber jar containing all the dependencies and execute the jar. Alternatively, you can execute each implementation as a stand-alone Scala project from your favourite IDE. 

## DeepLearning4j-based LSTM networks: 
The Long Short-term Memory (LSTM) network has been implemented to solve the classification problem. The following are prerequisites when working with DL4J:
- Java 1.8+ (64-bit only)
- Apache Maven for automated build and dependency manager
- IntelliJ IDEA or Eclipse IDE.

For more information on how to configure DeepLearning4j, please refer to https://deeplearning4j.org/. If everything is properly configured, you can create a uber jar containing all the dependencies and execute the jar. Alternatively, you can execute each implementation as a stand-alone Java project from your favourite IDE. 

## Citation request
If you reuse this implementation, please cite our paper: 

    @inproceedings{khan2018bigdata,
        title={A Two-stage Big Data Analytics Framework with Real World Applications using Spark ML and LSTM Networks},
        author={M. A., Khan; Karim, Md. Rezaul; Y. Kim },
        booktitle={Symmetry — Open Access Journal},
        year={2018}
    }

## Contributing
For any questions, feel free to open an issue or contact at rezaul.karim@rwth-aachen.de
