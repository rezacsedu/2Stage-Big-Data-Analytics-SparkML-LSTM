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
import org.apache.spark.ml.classification.DecisionTreeClassifier

object URLReputationDT {
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
    indexed.show()

    val guessedFraction = 0.1
    val newSampleDF = indexed.sample(true, guessedFraction).limit(1000)

    val splitSeed = 5043
    val Array(trainingData, testData) = newSampleDF.randomSplit(Array(0.80, 0.20), splitSeed)
    trainingData.cache()
    testData.cache()
    
    
    
    System.gc()

    val dt = new DecisionTreeClassifier()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    /*
    println("Preparing K-fold Cross Validation and Grid Search")
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.maxDepth, 3 :: 5 :: Nil) // :: 15 :: 20 :: 25 :: 30 :: Nil)
      .addGrid(rf.featureSubsetStrategy, "auto" :: "all" :: Nil)
      .addGrid(rf.impurity, "gini" :: "entropy" :: Nil)
      .addGrid(rf.maxBins, 3 :: 5 :: Nil) //10 :: 15 :: 25 :: 35 :: 45 :: Nil)
      .addGrid(rf.numTrees, 2 :: 3 :: Nil) // :: 100 :: Nil)
      .build()

    val cv = new CrossValidator()
      .setEstimator(rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    // ************************************************************
    println("Training model with Random Forest algorithm")
    // ************************************************************
*/
    val cvModel = dt.fit(trainingData)

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
  }
}