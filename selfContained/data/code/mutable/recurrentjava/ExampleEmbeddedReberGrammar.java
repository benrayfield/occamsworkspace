package mutable.recurrentjava;
import java.util.Random;

import immutable.lazycl.spec.Lazycl;
import immutable.rnn.RnnParams;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.trainer.Trainer;
import mutable.recurrentjava.util.NeuralNetworkHelper;
import mutable.util.Options;
import mutable.recurrentjava.datasets.EmbeddedReberGrammar;
import mutable.recurrentjava.datastructs.DataSet;

public class ExampleEmbeddedReberGrammar {
	public static void main(String[] args) throws Exception {
		
		Lazycl lz = Options.defaultLazycl();

		Random rng = new Random();
		
		DataSet data = new EmbeddedReberGrammar(lz, rng);
		
		int hiddenDimension = 12;
		int hiddenLayers = 1;
		float learningRate = .001f;
		float initParamsStdDev = .08f;

		int parallelSize = 1;
		Model nn = NeuralNetworkHelper.makeLstm(
				lz,
				parallelSize,
				data.inputDimension,
				hiddenDimension, hiddenLayers, 
				data.outputDimension, data.getModelOutputUnitToUse(), 
				initParamsStdDev, rng);
		
		int reportEveryNthEpoch = 10;
		int trainingEpochs = 1000;
		
		RnnParams p = new RnnParams().learnRate(learningRate);
		Trainer.train(lz, p, trainingEpochs, nn, data, reportEveryNthEpoch, rng);
		
		System.out.println("done.");
	}
}
