package org.rwth.URLReputation

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator

object URLReputationNB {
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

    val lsvc = new NaiveBayes()
    .setSmoothing(0.0001)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

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
    |            features|label|       rawPrediction|         probability|prediction|
+--------------------+-----+--------------------+--------------------+----------+
|(3231949,[1,3,4,5...|  0.0|[-635.64527252349...|[0.99999763270296...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[-602.73458293499...|[0.99999999999999...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[-604.04149257595...|[0.99999999999999...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[-690.38633636669...|[0.36059390589009...|       1.0|
|(3231949,[1,3,4,5...|  0.0|[-639.48543962809...|[0.99999999993474...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[-608.51606740753...|[0.99999999999997...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[-812.45307197178...|[0.99999992394735...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[-829.84761522716...|[0.99999999899229...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[-873.22209260529...|[1.0,9.9819321925...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[-838.17234299219...|[0.99998112203199...|       0.0|
+--------------------+-----+--------------------+--------------------+----------+
only showing top 10 rows

+-----+---------------+
|label|Predicted_label|
+-----+---------------+
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            1.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
|  0.0|            0.0|
+-----+---------------+
only showing top 10 rows

Accuracy: 0.924902048350481
Area under the precision-recall curve: 0.9242485041969178
Area under the receiver operating characteristic (ROC) curve: 0.924902048350481
Matthews correlation coefficient: 0.8485441585953132
Total Count: 3230
Correct: 2998
Wrong: 232
Ratio wrong: 0.0718266253869969
Ratio correct: 0.9281733746130031
Ratio true positive: 0.5777089783281734
Ratio false positive: 0.037461300309597524
Ratio true negative: 0.3504643962848297
Ratio false negative: 0.03436532507739938
     */
  }
}