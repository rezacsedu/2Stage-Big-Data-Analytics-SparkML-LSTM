package org.FIT.Arrhythmia

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature.PCA

object DecisionTrees {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("churn")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "C:/Users/admin-karim/Downloads/java/")
      .getOrCreate();

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

    val dt = new DecisionTreeClassifier().setSeed(12345)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************    
    val paramGrid = new ParamGridBuilder()
      .addGrid(dt.maxDepth, 3 :: 5 :: 10 :: 20 :: Nil) // :: 15 :: 20 :: 25 :: 30 :: Nil)
      .addGrid(dt.impurity, "gini" :: "entropy" :: Nil)
      .addGrid(dt.maxBins, 3 :: 5 :: 10 :: 20 :: Nil) //10 :: 15 :: 25 :: 35 :: 45 :: Nil)
      .build()

    val cv = new CrossValidator()
      .setEstimator(dt)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    // ************************************************************
    println("Training model with decision trees algorithm")
    // ************************************************************

    val cvModel = cv.fit(trainingData)

    //val model = classifier.fit(trainingData)
    val predictions = cvModel.transform(testData)
    predictions.show()

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
Training model with decision trees algorithm
+--------------------+-----+--------------------+--------------------+----------+
|            features|label|       rawPrediction|         probability|prediction|
+--------------------+-----+--------------------+--------------------+----------+
|[-149.95584654575...|   10|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|      10.0|
|[-104.25433077352...|    9|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       9.0|
|[-57.668763846291...|   10|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|      15.0|
|[-53.377125061440...|    1|[0.0,108.0,1.0,0....|[0.0,0.9152542372...|       1.0|
|[-51.289660705713...|    1|[0.0,14.0,2.0,0.0...|[0.0,0.6363636363...|       1.0|
|[-48.148578088983...|    1|[0.0,108.0,1.0,0....|[0.0,0.9152542372...|       1.0|
|[-47.953308925020...|    3|[0.0,0.0,0.0,9.0,...|[0.0,0.0,0.0,1.0,...|       3.0|
|[-40.073911472441...|    1|[0.0,5.0,0.0,0.0,...|[0.0,1.0,0.0,0.0,...|       1.0|
|[-39.341311711976...|    6|[0.0,108.0,1.0,0....|[0.0,0.9152542372...|       1.0|
|[-39.009684046704...|   10|[0.0,108.0,1.0,0....|[0.0,0.9152542372...|       1.0|
|[-37.990233369649...|    1|[0.0,14.0,2.0,0.0...|[0.0,0.6363636363...|       1.0|
|[-22.629726884179...|    1|[0.0,16.0,0.0,0.0...|[0.0,1.0,0.0,0.0,...|       1.0|
|[-18.363825656017...|    2|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|      16.0|
|[-18.080829448920...|    1|[0.0,108.0,1.0,0....|[0.0,0.9152542372...|       1.0|
|[-13.834169533516...|    4|[0.0,9.0,0.0,0.0,...|[0.0,0.75,0.0,0.0...|       1.0|
|[-2.3560516070688...|    1|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|       4.0|
|[7.52261962275276...|    3|[0.0,0.0,0.0,9.0,...|[0.0,0.0,0.0,1.0,...|       3.0|
|[8.66045828088360...|   14|[0.0,15.0,0.0,0.0...|[0.0,1.0,0.0,0.0,...|       1.0|
|[9.8367757396746,...|    1|[0.0,108.0,1.0,0....|[0.0,0.9152542372...|       1.0|
|[14.9954871845230...|    1|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|      10.0|
+--------------------+-----+--------------------+--------------------+----------+
only showing top 20 rows

Accuracy = 0.85440902897270863
Precision = 0.85440902897270863
Recall = 0.85440902897270863
F1 = 0.85440902897270863
Test Error = 0.1455909710272914
 * 
 * 
 */