package org.rwth.URLReputation

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator

object URLReputationSVM {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Reputation URL")
      .master("local[*]")
      //.config("spark.sql.warehouse.dir", "E:/Exp/")
      .getOrCreate();

    spark.conf.set("spark.sql.crossJoin.enabled", "true")

    import spark.implicits._

    var URLs = spark.read.format("libsvm").load("C:/Users/admin-karim/Desktop/url_svmlight.tar/url_svmlight/Day0.svm")
    //URLs.show()

    val indexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("labels")

    val indexed = indexer.fit(URLs).transform(URLs).drop("label").withColumnRenamed("labels", "label")
        
    val cols = indexed.columns.map(indexed(_)).reverse
    val reversedColDF = indexed.select(cols:_*)
    reversedColDF.show()    
    
    //val guessedFraction = 1.0
    //val newSampleDF = indexed.sample(true, guessedFraction).limit(5000)

    val splitSeed = 12345
    val Array(trainingData, testData) = indexed.randomSplit(Array(0.80, 0.20), splitSeed)
    trainingData.cache()
    testData.cache()
    
    //trainingData.coalesce(1).write.format("com.databricks.spark.csv").csv("results/train.csv")
    //testData.coalesce(1).write.format("com.databricks.spark.csv").csv("results/test.csv")   

    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    println("Preparing K-fold Cross Validation and Grid Search")
    val paramGrid = new ParamGridBuilder()
      .addGrid(lsvc.maxIter, 100 :: 1000 :: Nil) // :: 15 :: 20 :: 25 :: 30 :: Nil)
      .addGrid(lsvc.regParam, 0.001 :: 0.0001 :: Nil)
      .addGrid(lsvc.tol, 0.01 :: 0.001 :: Nil) //10 :: 15 :: 25 :: 35 :: 45 :: Nil)
      .build()

    val cv = new CrossValidator()
      .setEstimator(lsvc)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    // ************************************************************
    println("Training model with Linear SVM algorithm")

    val cvModel = lsvc.fit(trainingData)

    val predictions = cvModel.transform(testData)
    predictions.show(10)

    val result = predictions.select("label", "prediction")
    val resutDF = result.withColumnRenamed("prediction", "Predicted_label")
    resutDF.show(10)

    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy: " + accuracy)
    evaluator.explainParams()

    val predictionAndLabels = predictions
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1)
        .asInstanceOf[Double]))

    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    val areaUnderPR = metrics.areaUnderPR
    println("Area under the precision-recall curve: " + areaUnderPR)

    val areaUnderROC = metrics.areaUnderROC
    //val kkk = metrics.

    println("Area under the receiver operating characteristic (ROC) curve: " + areaUnderROC)

    val lp = predictions.select("label", "prediction")
    val counttotal = predictions.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" === $"prediction")).count()
    val ratioWrong = wrong.toDouble / counttotal.toDouble
    val ratioCorrect = correct.toDouble / counttotal.toDouble
    val tp = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count() / counttotal.toDouble
    val tn = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count() / counttotal.toDouble
    val fp = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count() / counttotal.toDouble
    val fn = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count() / counttotal.toDouble

    val MCC = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (fp + tn) * (tn + fn))
    println("Matthews correlation coefficient: " + MCC)

    println("Total Count: " + counttotal)
    println("Correct: " + correct)
    println("Wrong: " + wrong)
    println("Ratio wrong: " + ratioWrong)
    println("Ratio correct: " + ratioCorrect)
    println("Ratio true positive: " + tp)
    println("Ratio false positive: " + fp)
    println("Ratio true negative: " + tn)
    println("Ratio false negative: " + fn)
    
    
    /*
    |            features|label|       rawPrediction|prediction|
+--------------------+-----+--------------------+----------+
|(3231949,[1,3,4,5...|  0.0|[0.96053243107537...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[1.00359346935746...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[1.01528756500182...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[0.77553897765945...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[0.97488114710722...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[0.96843596854635...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[0.89497653542737...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[0.90286430973805...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[0.96690496729321...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[0.88576532916509...|       0.0|
+--------------------+-----+--------------------+----------+
only showing top 10 rows

+-----+---------------+
|label|Predicted_label|
+-----+---------------+
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
+-----+---------------+
only showing top 10 rows

Accuracy: 0.8466757576702307
Area under the precision-recall curve: 0.8809396531166327
Area under the receiver operating characteristic (ROC) curve: 0.8466757576702307
Matthews correlation coefficient: 0.7350873359923633
Total Count: 3230
Correct: 2823
Wrong: 407
Ratio wrong: 0.1260061919504644
Ratio correct: 0.8739938080495356
Ratio true positive: 0.5938080495356037
Ratio false positive: 0.021362229102167184
Ratio true negative: 0.2801857585139319
Ratio false negative: 0.10464396284829722
     */
  }
}