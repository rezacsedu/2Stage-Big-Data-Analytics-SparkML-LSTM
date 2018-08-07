package org.FIT.Arrhythmia

import org.apache.spark.sql.{ DataFrame, SQLContext, SparkSession }
import water.support.{ H2OFrameSupport, ModelMetricsSupport, SparkContextSupport }
import org.apache.spark.sql._
import org.apache.spark.sql.functions.year
import org.apache.spark.ml.feature.{ StringIndexer, VectorAssembler }
import org.apache.spark.ml.{ Pipeline, PipelineStage }
import scala.xml.persistent.SetStorage
import org.apache.spark.sql.functions._
import org.apache.spark.h2o._
import _root_.hex.FrameSplitter
import _root_.hex.{ ModelMetrics, ModelMetricsSupervised, ModelMetricsMultinomial }
import water.Key
import water.fvec.Frame
import _root_.hex.deeplearning.DeepLearning
import _root_.hex.deeplearning.DeepLearningModel
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters
import _root_.hex.deeplearning.DeepLearningModel.DeepLearningParameters.Activation
import java.io.File
import water.support.ModelSerializationSupport
import scala.reflect.api.materializeTypeTag
import org.apache.spark.sql.types._

object ArrhythmiaPredictionH2O extends SparkContextSupport with ModelMetricsSupport with H2OFrameSupport {
  /** Builds DeepLearning model. */
  def toCategorical(f: Frame, i: Int): Unit = {
    f.replace(i, f.vec(i).toCategoricalVec)
    f.update()
  }

  def ConfusionMatrix(model: DeepLearningModel, fr: Frame) = ModelMetrics.getFromDKV(model, fr).asInstanceOf[ModelMetricsMultinomial].cm()

  def buildDLModel(train: Frame, valid: Frame,
    epochs: Int = 10000, l1: Double = 0.01, l2: Double = 0.0,
    hidden: Array[Int] = Array[Int](32, 64, 128))(implicit h2oContext: H2OContext): DeepLearningModel = {
    import h2oContext.implicits._
    // Build a model
    val dlParams = new DeepLearningParameters()
    dlParams._train = train
    dlParams._valid = valid
    dlParams._response_column = "label"
    dlParams._epochs = epochs
    dlParams._nfolds = 10
    dlParams._l1 = l1
    dlParams._hidden = hidden
    dlParams._activation = Activation.TanhWithDropout
    dlParams._variable_importances = true

    // Create a job
    val dl = new DeepLearning(dlParams, water.Key.make("dlModel.hex"))
    dl.trainModel.get
  }
  
  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("churn")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "C:/Users/admin-karim/Downloads/java/")
      .getOrCreate();

    val arrhythmiaDF = spark.read.option("inferSchema", "false")
      .format("com.databricks.spark.csv")
      .option("inferSchema", "true")
      .option("header", "false")
      .load("data/arrhythmia.data").cache()

    arrhythmiaDF.show();

    //featureDF = featureDF.na.replace("*", Map("?" -> "0"))
    val dFwithoutSampleID = arrhythmiaDF.drop("_c10", "_c11", "_c12", "_c13", "_c14").withColumnRenamed("_c279", "label")
    dFwithoutSampleID.printSchema()
    //val dFwithoutSampleID = featureDF.withColumnRenamed("_c279", "label")
    println(dFwithoutSampleID.columns.length)

    implicit val h2oContext = H2OContext.getOrCreate(spark.sparkContext)
    import h2oContext.implicits._
    implicit val sqlContext = SparkSession.builder().getOrCreate().sqlContext
    import sqlContext.implicits._

    val H2ODF: H2OFrame = dFwithoutSampleID.orderBy(rand())
    H2ODF.types.zipWithIndex.foreach(c => if (c._1.toInt == 2) toCategorical(H2ODF, c._2))
    val sf = new FrameSplitter(H2ODF, Array(0.6, 0.2), Array("train.hex", "valid.hex", "test.hex").map(Key.make[Frame](_)), null)
    water.H2O.submitTask(sf)

    val splits = sf.getResult
    val (train, valid, test) = (splits(0), splits(1), splits(2))

    // Build a model
    toCategorical(train, 274)
    toCategorical(valid, 274)
    toCategorical(test, 274)

    val dlModel = buildDLModel(train, valid)

    val result = dlModel.score(test)('predict)
    val predictionsFromModel = h2oContext.asRDD[DoubleHolder](result).take(10).map(_.result.getOrElse("NaN"))
    println(predictionsFromModel.mkString("\n===> Model predictions: ", ", ", ", ...\n"))

    val output = dlModel._output

    println("Classfication accuracy: " + (1 - dlModel.classification_error()) * 100 + " %")
    println(dlModel.mean_per_class_error())

    println("Confusion matrix on test set: ")
    println(ConfusionMatrix(dlModel, test))

    result.add("actual", test.vec("label"))
    val predict_actualDF = h2oContext.asDataFrame(result)
    predict_actualDF.show()

    predict_actualDF.groupBy("actual", "predict").count.show

    dlModel.score(test)('predict)

    // Shutdown Spark cluster and H2O
    h2oContext.stop(stopSparkContext = true)
    spark.stop()
  }
}

/*
===> Model predictions: 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 11.0, 0.0, 8.0, ...

Classfication accuracy: 66.66666666666667 %
Confusion matrix on test set: 
[43.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
[5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[4.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
[2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

+-------+------+
|predict|actual|
+-------+------+
|      1|     1|
|      2|     2|
|      2|     1|
|      1|     1|
|      1|     1|
|      2|    10|
|      1|     1|
|     16|     1|
|      1|    16|
|     10|    10|
|      1|     1|
|      1|     1|
|      4|     4|
|      1|     7|
|      1|     1|
|      1|     1|
|      1|     1|
|     10|    10|
|      1|     1|
|      1|     1|
+-------+------+
only showing top 20 rows

+------+-------+-----+
|actual|predict|count|
+------+-------+-----+
|     7|      1|    1|
|     4|      1|    1|
|    15|      2|    2|
|     1|      4|    2|
|     5|      1|    4|
|     5|      5|    1|
|     1|      1|   43|
|     6|      6|    1|
|    16|      1|    2|
|    10|      2|    1|
|     4|      4|    1|
|     3|      3|    2|
|     2|      2|    5|
|     1|      2|    2|
|     2|     14|    1|
|    10|      1|    2|
|     2|     16|    1|
|     1|      5|    1|
|     1|     16|    2|
|     2|      1|    5|
+------+-------+-----+
only showing top 20 rows
*/
