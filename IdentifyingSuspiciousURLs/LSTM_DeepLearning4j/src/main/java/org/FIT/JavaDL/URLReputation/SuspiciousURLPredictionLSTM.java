package org.FIT.JavaDL.Arrthimya;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/*
 * @Author: Md. Rezaul Karim, 7th August 2018
 * Research Scientist, Fraunhofer FIT, Germany
 */

public class ArrthimyaPredictionLSTM {
	private static final Logger log = LoggerFactory.getLogger(ArrthimyaPredictionLSTM.class);
	//static int batchSizePerWorker = 16;
	static int numEpochs = 1000;

	private static DataSetIterator readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses) throws IOException, InterruptedException {
		RecordReader rr = new CSVRecordReader();
		File input = new File(csvFileClasspath);
		rr.initialize(new FileSplit(input));
		DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
		return iterator;
	}

	public static void main(String[] args) throws Exception {	

		// Show data paths
		String trainPath = "data/train.csv";
		String testPath = "data//test.csv";	
		
		// ----------------------------------
		// Preparing training and test set. 	
		int labelIndex = 274;			
		int numClasses = 17; 
		int batchSize = 32; 
		
		// This dataset is used for training 
		DataSetIterator trainingDataIt = readCSVDataset(trainPath, batchSize, labelIndex, numClasses);

		// This is the data we want to classify
		DataSetIterator testDataIt = readCSVDataset(testPath, batchSize, labelIndex, numClasses);		
		
		// ----------------------------------
		// Network hyperparameters
		int seed = 12345;
		int numInputs = labelIndex;
		int numOutputs = numClasses;		
		int numHiddenNodes = 256;
		
		// Create network configuration and conduct network training
        MultiLayerConfiguration LSTMconf = new NeuralNetConfiguration.Builder()
            .seed(seed)    //Random number generator seed for improved repeatability. Optional.
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.0001))
            .list()
            .layer(0, new LSTM.Builder()
            			.nIn(numInputs)
            			.nOut(numHiddenNodes)
            			.activation(Activation.RELU)
            			.build())
            .layer(1, new LSTM.Builder()
            			.nIn(numHiddenNodes)
            			.nOut(numHiddenNodes)
            			.activation(Activation.RELU)
            			.build())
            .layer(2, new LSTM.Builder()
            			.nIn(numHiddenNodes)
            			.nOut(numHiddenNodes)
            			.activation(Activation.RELU)
            			.build())
            .layer(3, new RnnOutputLayer.Builder()
            			.activation(Activation.SOFTMAX)
            			.lossFunction(LossFunction.MCXENT)
            			.nIn(numHiddenNodes)
            			.nOut(numOutputs)
            			.build())
            .pretrain(false).backprop(true).build();

        // Create and initialize multilayer network 
		MultiLayerNetwork model = new MultiLayerNetwork(LSTMconf);
        model.init();
        
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = model.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
        	//DataSet next = trainingDataIt.next();
            model.fit(trainingDataIt);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(17); //create an evaluation object with 10 possible classes
        while(testDataIt.hasNext()){
            DataSet next = testDataIt.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
	}
}


/*

==========================Scores========================================
# of classes:    17
Accuracy:        0.9488
Precision:       0.9604	(10 classes excluded from average)
Recall:          0.9678	(7 classes excluded from average)
F1 Score:        0.9617	(10 classes excluded from average)
Precision, recall & F1: macro-averaged (equally weighted avg. of 17 classes)
========================================================================
22:11:38.041 [main] INFO com.packt.JavaDL.Arrthimya.ArrthimyaPredictionLSTM - ****************Example finished********************

*/