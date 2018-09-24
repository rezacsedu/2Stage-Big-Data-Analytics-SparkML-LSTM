package org.rwth.URLReputation

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
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.classification.LogisticRegression

object URLReputationLR2 {
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

    val lr = new LogisticRegression()
    .setMaxIter(10000)
    .setRegParam(0.001)
    .setTol(0.001)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    println("Preparing K-fold Cross Validation and Grid Search")
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.maxIter, 100 :: 1000 :: Nil) // :: 15 :: 20 :: 25 :: 30 :: Nil)
      .addGrid(lr.regParam, 0.001 :: 0.0001 :: Nil)
      .addGrid(lr.tol, 0.01 :: 0.001 :: Nil) //10 :: 15 :: 25 :: 35 :: 45 :: Nil)
      .build()

    val cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    // ************************************************************
    println("Training model with Random Forest algorithm")

    val cvModel = lr.fit(trainingData)

    val predictions = cvModel.transform(testData)
    predictions.show(10)

    val result = predictions.select("label", "prediction", "probability")
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
     *                                                                                 +--------------------+-----+--------------------+--------------------+----------+
|            features|label|       rawPrediction|         probability|prediction|
+--------------------+-----+--------------------+--------------------+----------+
|(3231949,[1,3,4,5...|  0.0|[3.27501770732767...|[0.96356175629735...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[3.92848718535972...|[0.98070616332516...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[5.15098357996895...|[0.99423967028870...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[0.49936505284902...|[0.62231010466619...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[3.83814712452765...|[0.97892045243246...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[3.96806744015681...|[0.98144100704751...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[1.28988777363898...|[0.78412819309580...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[5.05373360555550...|[0.99365506709979...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[6.76827377010483...|[0.99885164269391...|       0.0|
|(3231949,[1,3,4,5...|  0.0|[3.54186949504432...|[0.97185589162336...|       0.0|
+--------------------+-----+--------------------+--------------------+----------+
only showing top 10 rows

+-----+---------------+--------------------+
|label|Predicted_label|         probability|
+-----+---------------+--------------------+
|  0.0|            0.0|[0.96356175629735...|
|  0.0|            0.0|[0.98070616332516...|
|  0.0|            0.0|[0.99423967028870...|
|  0.0|            0.0|[0.62231010466619...|
|  0.0|            0.0|[0.97892045243246...|
|  0.0|            0.0|[0.98144100704751...|
|  0.0|            0.0|[0.78412819309580...|
|  0.0|            0.0|[0.99365506709979...|
|  0.0|            0.0|[0.99885164269391...|
|  0.0|            0.0|[0.97185589162336...|
+-----+---------------+--------------------+
only showing top 10 rows

Accuracy: 0.9502089405755269
Area under the precision-recall curve: 0.9553596420423565
Area under the receiver operating characteristic (ROC) curve: 0.9502089405755269
Matthews correlation coefficient: 0.9055804378063441
Total Count: 3230
Correct: 3086
Wrong: 144
Ratio wrong: 0.04458204334365325
Ratio correct: 0.9554179566563468
Ratio true positive: 0.5984520123839009
Ratio false positive: 0.01671826625386997
Ratio true negative: 0.3569659442724458
Ratio false negative: 0.02786377708978328

     * 
     */
  }
}