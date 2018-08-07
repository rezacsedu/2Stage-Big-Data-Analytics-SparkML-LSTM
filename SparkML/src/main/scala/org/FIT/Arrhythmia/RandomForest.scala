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

object RandomForest {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("churn")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "E:/Exp/")
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

    var featureDF = spark.sql("SELECT * FROM arrhythmia")

    featureDF = featureDF.na.replace("*", Map("?" -> "0"))
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

    val finalDF = featureDF.withColumnRenamed("_c279", "label")
    finalDF.show()
    
    finalDF.createOrReplaceTempView("kutta")
    spark.sql("SELECT DISTINCT label from kutta").show()
    
    println(finalDF.columns.length)
    val splitSeed = 5043
    val Array(trainingData, testData) = finalDF.randomSplit(Array(0.80, 0.20), splitSeed)
    //trainingData.write.format("com.databricks.spark.csv").csv("results/train.csv")
    //testData.write.format("com.databricks.spark.csv").csv("results/test.csv")    

    //val splitSeed = 5043
    //val Array(trainingData, testData) = finalDF.randomSplit(Array(0.80, 0.20), splitSeed)

    val rf = new RandomForestClassifier().setSeed(12345)

    // Building the Pipeline for transformations and predictor
    println("Building Machine Learning pipeline")
    val pipeline = new Pipeline().setStages(Array(assembler, rf))

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************    
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, 3 :: 5 :: 10 :: Nil) // :: 15 :: 20 :: 25 :: 30 :: Nil)
      .addGrid(rf.featureSubsetStrategy, "auto" :: "all" :: Nil)
      .addGrid(rf.impurity, "gini" :: "entropy" :: Nil)
      .addGrid(rf.maxBins, 3 :: 5 :: 10 :: Nil) //10 :: 15 :: 25 :: 35 :: 45 :: Nil)
      .addGrid(rf.numTrees, 3 :: 5 :: 10 :: Nil) // :: 100 :: Nil)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    // ************************************************************
    println("Training model with Random Forest algorithm")
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
 * Accuracy = 0.87598233602084693
Precision = 0.8598233602084693
Recall = 0.8698233602084693
F1 = 0.8698233602084693
Test Error = 0.13017663979153066

*/
