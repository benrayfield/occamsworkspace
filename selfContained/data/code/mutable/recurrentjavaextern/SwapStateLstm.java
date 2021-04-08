package mutable.recurrentjavaextern;
import static mutable.util.Lg.lg;

import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;

import immutable.lazycl.spec.Lazycl;
import immutable.recurrentjava.flop.unary.SigmoidUnit;
import immutable.rnn.recurrentjava.RjLearnStep;
import immutable.util.Blob;
import immutable.util.BlobUtil;
import immutable.util.MathUtil;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.recurrentjava.autodiff.CpuGraph;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.loss.LossSumOfSquares;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.util.NeuralNetworkHelper;
import mutable.recurrentjava.model.*;
import mutable.recurrentjava.trainer.Trainer;

/** has 2 node states ({out,sum} and {out,sum}) for prediction and training separately
so the one used for prediction can continue smoothly while alternating training and prediction.
*/
@Deprecated //use immutable.lstm.Lstm instead
public class SwapStateLstm{
	
	protected Model neuralnet;
	
	/** swaps between 2 copies of mutable.recurrentjava.model.GruLayer.context,
	one for training and one for prediction. The prediction one is never reset
	and is used only during prediction but a different one is used during training.
	Its swapped in this.neuralnet many times per second.
	*/
	protected Matrix swappedStateHidden;
	
	protected Matrix swappedStateCell;
	
	/** true if swappedState is the prediction Matrix and neuralnet contains the training matrix,
	else that but swapped.
	*/
	protected boolean swappedStateIsPrediction = true;
	
	/** never reset prediction state, but reset training state at start of each DataSequence.
	Training creates its own Graphs.
	FIXME?? is this graph being used at all? I'm not backproping in it. swappedState is probably all thats needed
	and could use a new Graph every prediction to get the same effect???
	*/
	protected final Graph predictionGraph;
	
	public final Lazycl lz;

	/** These lists may contain the same InsOuts multiple times or not,
	such as if its made of a List<InsOuts> and a timewindow over various partially overlapping ranges. 
	*
	public List<List<InsOuts>> trainingData = new ArrayList();
	*/
	
	public final int ins, hiddens, outs;
	
	public SwapStateLstm(Lazycl lz, int ins, int hiddens, int outs){
		this.lz = lz;
		predictionGraph = new CpuGraph(lz, false);
		this.ins = ins;
		this.hiddens = hiddens;
		this.outs = outs;
		float initParamsStdDev = .08f;
		int parallelSize = 1;
		neuralnet = NeuralNetworkHelper.makeLstm(
			lz,
			parallelSize,
			ins,
			hiddens, 1, 
			outs, new SigmoidUnit(), 
			initParamsStdDev, Rand.strongRand);
		neuralnet.resetState();
		swap();
		neuralnet.resetState();
		
	}
	
	/** changes swappedStateIsPrediction and swappedState */
	public void swap(){
		LstmLayer l = (LstmLayer)((NeuralNetwork)neuralnet).layers.get(0);
		Matrix temp = l.hiddenContext;
		l.hiddenContext = swappedStateHidden;
		swappedStateHidden = temp;
		temp = l.cellContext;
		l.cellContext = swappedStateCell;
		swappedStateCell = temp;
		swappedStateIsPrediction = !swappedStateIsPrediction;
	}
	
	/** FIXME verify recurrentjava can handle more than 1 trainingvector
	at a timestep (like Im going to do when optimizing for opencl).
	*/
	public static Matrix toMatrix(Lazycl lz, float[][] d){
		int rows = d.length, cols = d[0].length;
		Matrix m = new Matrix(lz, rows, cols);
		/*int offset = 0;
		for(int y=0; y<m.rows; y++){
			System.arraycopy(d[y], 0, m.buf("w"), offset, m.cols);
			offset += m.cols;
		}*/
		m.put("w",lz.wrapb(MathUtil.array2dTo1d(d)));
		return m;
	}
	
	/** FIXME verify recurrentjava can handle more than 1 trainingvector
	at a timestep (like Im going to do when optimizing for opencl).
	*/
	public static float[][] toDoubles(Matrix m){
		float[][] ret = new float[m.rows][m.cols];
		int offset = 0;
		Blob w = m.get("w");
		for(int y=0; y<m.rows; y++){
			BlobUtil.arraycopy(w, offset, ret[y], 0, m.cols);
			//System.arraycopy(m.buf("w"), offset, ret[y], 0, m.cols);
			offset += m.cols;
		}
		return ret;
	}
	
	public static DataStep toDatastep(RjLearnStep io){
		throw new Error("TODO");
		/*double[] empty = new double[0];
		DataStep d = new DataStep(empty, empty);
		d.input = toMatrix(io.ins);
		d.targetOutput = toMatrix(io.outs);
		return d;
		*/
	}
	
	public static DataSequence toDataSequence(List<RjLearnStep> sequence){
		return new DataSequence(Arrays.asList(
			(DataStep[])sequence.stream().map(SwapStateLstm::toDatastep).toArray()
		));
	}
	
	public void predict(){
		throw new Error("TODO should I just hardcode it the swapstate way Ive been doing?"
			+" I want this general enough I dont have to redesign it when I change to interactive"
			+" (such as game playing, mouseai, etc) instead of recorded datasets."
			+" Make the neuralnet immutable and just use recurrentjava as a transition func."
			+" That way when I upgrade to opencl I can use the same javaclass and just replace"
			+" the body of the transition func."
			+" How about that immutable treemap I was thinking of using before I finish occamsfuncer?"
			+" I dont want to get into calling the compilers (javassist, jdk, openjdk) on a code string"
			+" in that map for now (occamsfuncer would sandbox and compile to that and opencl)."
			+" I could use JsonDS maps which are immutable but inefficient to forkEdit"
			+" cuz not treemap.");
	}
	
	public void learn(float learnRate, List<RjLearnStep> sequence){
		throw new Error("TODO");
		/*if(!sequence.isEmpty() && sequence.get(0).ins.length > 1)
			throw new Error("TODO verify recurrentjava can learn multiple training vecs in a timestep (like Im planning to do in opencl upgrade either way) before doing that here (else will only do it in opencl).");
		DataSequence ds = toDataSequence(sequence);
		swap();
		if(!swappedStateIsPrediction) throw new Error("training matrix should be in neuralnet but is in swappedState");
		//double learnRate = 0.001;
		//double learnRate = 0.0001;
		//double learnRate = 0.003;
		boolean applyTraining = true;
		Loss lossTraining = new LossSumOfSquares(); //for waves such as mouse
		//Loss lossTraining = new LossSoftmax(); //for text etc
		Loss lossReporting = lossTraining;
		try{
			Trainer.pass(learnRate, neuralnet, Arrays.asList(ds), applyTraining, lossTraining, lossReporting);
		}catch(Exception e){
			throw new Error(e);
		}
		swap();
		
		if(swappedStateIsPrediction) throw new Error("prediction matrix should be in neuralnet but is in swappedState");
		*/
	}

}

