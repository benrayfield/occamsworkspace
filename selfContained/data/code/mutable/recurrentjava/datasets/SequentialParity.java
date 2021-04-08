package mutable.recurrentjava.datasets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.lazycl.spec.Lazycl;
import immutable.recurrentjava.flop.unary.SigmoidUnit;
import mutable.recurrentjava.loss.LossMultiDimensionalBinary;
import mutable.recurrentjava.loss.LossSumOfSquares;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataSet;
import mutable.recurrentjava.datastructs.DataStep;


public class SequentialParity extends DataSet{
	
	public final Lazycl lz;

	public SequentialParity(Lazycl lz, Random r, int total_sequences, int max_sequence_length_train, int max_sequence_length_test) {
		this.lz = lz;
		inputDimension = 1;
		outputDimension = 1;
		lossTraining = new LossSumOfSquares();
		lossReporting = new LossMultiDimensionalBinary();
		training = generateSequences(lz(), r, total_sequences, max_sequence_length_train);
		
		//training.addAll(generateSequences(r, total_sequences, max_sequence_length_test));
		
		validation = generateSequences(lz(), r, total_sequences, max_sequence_length_train);
		testing = generateSequences(lz(), r, total_sequences, max_sequence_length_test);
	}
	
	public Lazycl lz(){
		return lz;
	}
	
	private static List<DataSequence> generateSequences(Lazycl lz, Random r, int total_sequences, int max_sequence_length) {
		List<DataSequence> result = new ArrayList<>();;
		for (int s = 0; s < total_sequences; s++) {
			DataSequence sequence = new DataSequence();
			int tot = 0;
			int tempSequenceLength = r.nextInt(max_sequence_length) + 1;
			for (int t = 0; t < tempSequenceLength; t++) {
				DataStep step = new DataStep(lz);
				float[] input = {0f};
				
				if (r.nextDouble() < 0.5) {
					input[0] = 1f;
					tot++;
				}
				step.input = new Matrix(lz, input);
				
				float[] targetOutput = null;
				if (t == tempSequenceLength - 1) {
					targetOutput = new float[1];
					targetOutput[0] = tot%2;
					step.targetOutput = new Matrix(lz, targetOutput);
				}
				sequence.steps.add(step);
			}
			result.add(sequence);
		}
		return result;
	}

	@Override
	public void DisplayReport(Model model, Random rng) throws Exception {
		// TODO Auto-generated method stub
	}

	@Override
	public Unaflop getModelOutputUnitToUse() {
		return new SigmoidUnit();
	}
}
