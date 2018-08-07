package org.FIT.Arrhythmia

import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.PCA

object LogisticRegressionClassifier {
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

    val lr = new LogisticRegression()
        
    val evaluator = new MulticlassClassificationEvaluator()
                        .setLabelCol("label")
                        .setPredictionCol("prediction")   

    // ***********************************************************
    println("Preparing K-fold Cross Validation and Grid Search")
    // ***********************************************************  
    /*
          .setFeaturesCol("features")
      .setLabelCol("indexedLabel")
      .setRegParam(params.regParam)
      .setElasticNetParam(params.elasticNetParam)
      .setMaxIter(params.maxIter)
      .setTol(params.tol)
      .setFitIntercept(params.fitIntercept)
      * 
      */
      
	val paramGrid = new ParamGridBuilder()
    .addGrid(lr.regParam, 0.1 :: 0.01 :: Nil)
    .addGrid(lr.maxIter, 10 :: 50 :: 100 :: 10000 :: Nil)
    .addGrid(lr.tol, 0.1 :: 0.01 :: 0.001 :: Nil)
    .build()

    val cv = new CrossValidator()
      .setEstimator(lr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

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
 *  * Preparing K-fold Cross Validation and Grid Search
Training model with Logistic regression algorithm
+--------------------+-----+--------------------+--------------------+----------+
|            features|label|       rawPrediction|         probability|prediction|
+--------------------+-----+--------------------+--------------------+----------+
|[-149.95584654575...|   10|[-4.8946079353081...|[4.76476124346340...|      10.0|
|[-104.25433077352...|    9|[-4.7922406446903...|[2.03534345512659...|       1.0|
|[-57.668763846291...|   10|[-4.6900479284940...|[2.52050948876797...|      10.0|
|[-53.377125061440...|    1|[-4.8121491982859...|[4.07753956389963...|       6.0|
|[-51.289660705713...|    1|[-4.6813329284192...|[1.69711081513775...|       1.0|
|[-48.148578088983...|    1|[-4.5854734576884...|[1.90717781534823...|       1.0|
|[-47.953308925020...|    3|[-4.8768529803121...|[6.85247455924077...|       3.0|
|[-40.073911472441...|    1|[-4.8317883851269...|[2.28657201200378...|      16.0|
|[-39.341311711976...|    6|[-5.0561979196995...|[1.36767087710177...|       6.0|
|[-39.009684046704...|   10|[-4.7549792248488...|[1.48154065628169...|      10.0|
|[-37.990233369649...|    1|[-4.5372758030996...|[9.12520636863799...|       1.0|
|[-22.629726884179...|    1|[-4.7705305370615...|[1.68358657928057...|       1.0|
|[-18.363825656017...|    2|[-4.7886306569392...|[9.80326265699974...|       2.0|
|[-18.080829448920...|    1|[-4.7044994436233...|[7.53798630806703...|       1.0|
|[-13.834169533516...|    4|[-4.5739856232036...|[1.96708852866269...|       2.0|
|[-2.3560516070688...|    1|[-4.7443898970392...|[1.54630113082309...|       1.0|
|[7.52261962275276...|    3|[-4.9431806185510...|[2.70655780603233...|       3.0|
|[8.66045828088360...|   14|[-4.8860590984415...|[1.14258297384001...|       2.0|
|[9.8367757396746,...|    1|[-4.5511319208121...|[1.48536005704231...|      10.0|
|[14.9954871845230...|    1|[-4.7172499065558...|[9.96978027848707...|       1.0|
+--------------------+-----+--------------------+--------------------+----------+
only showing top 20 rows

Accuracy = 0.8259510456067128
Precision = 0.8259510456067128
Recall = 0.8259510456067128
F1 = 0.8259510456067128
Test Error = 0.1740489543932872
*/