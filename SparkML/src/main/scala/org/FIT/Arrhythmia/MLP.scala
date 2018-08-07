package org.FIT.Arrhythmia

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.PCA

object MLP {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("churn")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "C:/Users/admin-karim/Downloads/java/")
      .getOrCreate();

    //spark.conf()    
    spark.conf.set("spark.sql.crossJoin.enabled", "true")

    import spark.implicits._

    val arrhythmiaDF = spark.read.option("inferSchema", "false")
      .format("com.databricks.spark.csv")
      .option("inferSchema", "true")
      .option("header", "false")
      .load("data/arrhythmia.data")

    arrhythmiaDF.show();
    arrhythmiaDF.createOrReplaceTempView("arrhythmia")

    val labelDF = spark.sql("SELECT _c279 FROM arrhythmia")
    labelDF.show()
    labelDF.createOrReplaceTempView("label")
    spark.sql("SELECT COUNT(DISTINCT *) FROM label").show(30)

    var featureDF = spark.sql("SELECT * FROM arrhythmia")

    //featureDF = featureDF.na.replace("*", Map("?" -> "0"))
    featureDF = featureDF.drop("_c10", "_c11", "_c12", "_c13", "_c14")
    /*

    featureDF = featureDF.withColumn("_c10", featureDF("_c10").cast(DoubleType))
      .withColumn("_c11", featureDF("_c11").cast(DoubleType))
      .withColumn("_c12", featureDF("_c12").cast(DoubleType))
      .withColumn("_c13", featureDF("_c13").cast(DoubleType))
      .withColumn("_c14", featureDF("_c14").cast(DoubleType))
      * 
      */

    featureDF.show()

    val colNames = featureDF.columns.dropRight(1)

    val assembler = new VectorAssembler()
      .setInputCols(colNames)
      .setOutputCol("features")

    val assembleDF = assembler.transform(featureDF).withColumnRenamed("_c279", "label")
    assembleDF.show()

    //val finalDF = featureDF.withColumnRenamed("_c279", "label")
    //finalDF.show()

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(50)
      .fit(assembleDF)

    val resultDF = pca.transform(assembleDF).select("pcaFeatures", "label").withColumnRenamed("pcaFeatures", "features")  
    
    val splitSeed = 5043
    val Array(trainingData, testData) = resultDF.randomSplit(Array(0.80, 0.20), splitSeed)

    val layers = Array[Int](50, 16, 32, 17)
    // create the trainer and set its parameters
    val mlp = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setTol(1E-4)
      .setMaxIter(1000)
    // ************************************************************
    println("Training model with Random Forest algorithm")
    // ************************************************************

    val cvModel = mlp.fit(trainingData)

    //val model = classifier.fit(trainingData)
    val predictions = cvModel.transform(testData)
    predictions.show()

    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction")
    val evaluator1 = evaluator.setMetricName("accuracy")
    val evaluator2 = evaluator.setMetricName("weightedPrecision")
    val evaluator3 = evaluator.setMetricName("weightedRecall")
    val evaluator4 = evaluator.setMetricName("f1")

    // compute the classification accuracy, precision, recall, f1 measure and error on test data.
    val accuracy = evaluator1.evaluate(predictions)
    val precision = evaluator2.evaluate(predictions)
    val recall = evaluator3.evaluate(predictions)
    val f1 = evaluator4.evaluate(predictions)

    // Print the performance metrics
    println("Accuracy = " + accuracy);
    println("Precision = " + precision)
    println("Recall = " + recall)
    println("F1 = " + f1)
    println(s"Test Error = ${1 - accuracy}")
  }
}


/*
 * 
 * Preparing K-fold Cross Validation and Grid Search
Training model with Multilayer perceptron algorithm
+--------------------+-----+--------------------+--------------------+----------+
|            features|label|       rawPrediction|         probability|prediction|
+--------------------+-----+--------------------+--------------------+----------+
|[-149.95584654575...|   10|[-38.015047322834...|[4.62664664680469...|      10.0|
|[-104.25433077352...|    9|[-48.433307604874...|[1.75192249740705...|       9.0|
|[-57.668763846291...|   10|[-42.903426389360...|[1.25438830443045...|      10.0|
|[-53.377125061440...|    1|[-46.814112530396...|[2.49283038016444...|       4.0|
|[-51.289660705713...|    1|[-52.696737351828...|[1.85031202276840...|       1.0|
|[-48.148578088983...|    1|[-58.114015117577...|[3.53453482305559...|       1.0|
|[-47.953308925020...|    3|[-49.843332036555...|[1.59033124777095...|       3.0|
|[-40.073911472441...|    1|[-50.856913843244...|[2.02049080563416...|       6.0|
|[-39.341311711976...|    6|[-50.856913843244...|[2.02049080563416...|       6.0|
|[-39.009684046704...|   10|[-52.000327559047...|[1.34944855461621...|       4.0|
|[-37.990233369649...|    1|[-52.696737351828...|[1.85031202276840...|       1.0|
|[-22.629726884179...|    1|[-52.000327559047...|[1.34944855461621...|       4.0|
|[-18.363825656017...|    2|[-52.696737351828...|[1.85031202276840...|       1.0|
|[-18.080829448920...|    1|[-58.114015117577...|[3.53453482305559...|       1.0|
|[-13.834169533516...|    4|[-48.986961893330...|[2.81228196040235...|       4.0|
|[-2.3560516070688...|    1|[-57.863988155436...|[4.20606802069147...|       1.0|
|[7.52261962275276...|    3|[-49.843332036555...|[1.59033124776918...|       3.0|
|[8.66045828088360...|   14|[-50.856913849633...|[2.02049078192166...|       6.0|
|[9.8367757396746,...|    1|[-52.696737351828...|[1.85031202276877...|       1.0|
|[14.9954871845230...|    1|[-58.114015117577...|[3.53453482305559...|       1.0|
+--------------------+-----+--------------------+--------------------+----------+
only showing top 20 rows

Accuracy = 0.8889097352511987
Precision = 0.8889097352511987
Recall = 0.8889097352511987
F1 = 0.8889097352511987
Test Error = 0.1110902647488013
 * 
 * 
 */