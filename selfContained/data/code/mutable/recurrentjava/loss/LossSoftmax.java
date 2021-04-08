package mutable.recurrentjava.loss;
import java.util.ArrayList;
import java.util.List;

import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.autodiff.CpuGraph;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.matrix.MatrixCache;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.util.Util;


public class LossSoftmax implements Loss {
	
	public final Lazycl lz;
	
	public LossSoftmax(Lazycl lz){
		this.lz = lz;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public void backward(Matrix logProbs, Matrix targetOutput) {
		int targetIndex = getTargetIndex(targetOutput);
		Matrix probs = getSoftmaxProbs(lz, logProbs, 1f);
		MatrixCache logProbsDw = logProbs.cache("dw");
		Blob probsW = probs.get("w");
		int probsWSize = probsW.fsizeIntElseThrow();
		for (int i = 0; i < probsWSize; i++){
			logProbsDw.put(i, probsW.f(i));
		}
		logProbsDw.putPlus(targetIndex, -1f);
	}

	@Override
	public float measure(Matrix logprobs, Matrix targetOutput){
		int targetIndex = getTargetIndex(targetOutput);
		Matrix probs = getSoftmaxProbs(lz, logprobs, 1f);
		Blob probsW = probs.get("w");
		float cost = (float) -Math.log(probsW.f(targetIndex));
		return cost;
	}

	public static float calculateMedianPerplexity(Lazycl lz, Model model, List<DataSequence> sequences){
		float temperature = 1f;
		List<Float> ppls = new ArrayList<>();
		for (DataSequence seq : sequences) {
			float n = 0;
			float neglog2ppl = 0;
			
			//Graph g = new Graph(false);
			Graph g = new CpuGraph(lz, false); //FIXME? if using OpenclGraph other places, dont create a new one, reuse same DependnetBuilder.
			model.resetState();
			for (DataStep step : seq.steps) {
				Matrix logprobs = model.forward(step.input, g);
				Matrix probs = getSoftmaxProbs(lz, logprobs, temperature);
				Blob probsW = probs.get("w");
				int targetIndex = getTargetIndex(step.targetOutput);
				float probOfCorrect = probsW.f(targetIndex);
				float log2prob = (float)(Math.log(probOfCorrect)/Math.log(2)); //change-of-base
				neglog2ppl += -log2prob;
				n += 1;
			}
			
			n -= 1; //don't count first symbol of sentence
			float ppl = (float)Math.pow(2, (neglog2ppl/(n-1)));
			ppls.add(ppl);
		}
		return Util.median(ppls);
	}
	
	public static Matrix getSoftmaxProbs(Lazycl lz, Matrix logprobs, float temperature){	
		Matrix probs = new Matrix(lz, logprobs.size);
		MatrixCache logprobsW = logprobs.cache("w");
		if (temperature != 1.0) {
			for (int i = 0; i < logprobs.size; i++) {
				logprobsW.putDivide(i, temperature);
			}
		}
		float maxval = Float.NEGATIVE_INFINITY;
		MatrixCache probsW = probs.cache("w");
		for (int i = 0; i < logprobs.size; i++) {
			if (logprobsW.get(i) > maxval) {
				maxval = logprobsW.get(i);
			}
		}
		float sum = 0;
		for (int i = 0; i < logprobs.size; i++) {
			probsW.put(i, (float)Math.exp(logprobsW.get(i) - maxval)); //all inputs to exp() are non-positive
			sum += probsW.get(i);
		}
		for (int i = 0; i < probsW.size(); i++) {
			probsW.putDivide(i, sum);
		}
		MatrixCache.closeAll(probsW, logprobsW);
		return probs;
	}

	private static int getTargetIndex(Matrix targetOutput){
		Blob targetOutputW = targetOutput.get("w");
		for (int i = 0; i < targetOutput.size; i++) {
			if (targetOutputW.f(i) == 1f) {
				return i;
			}
		}
		throw new Error("no target index selected");
	}
}
