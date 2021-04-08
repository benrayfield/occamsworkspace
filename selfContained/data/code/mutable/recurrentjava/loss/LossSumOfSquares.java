package mutable.recurrentjava.loss;
import immutable.util.Blob;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.matrix.MatrixCache;

public class LossSumOfSquares implements Loss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public void backward(Matrix actualOutput, Matrix targetOutput){
		MatrixCache actualOutputDw = actualOutput.cache("dw");
		Blob actualOutputW = actualOutput.get("w");
		Blob targetOutputW = targetOutput.get("w");
		for (int i = 0; i < targetOutput.size; i++){
			float errDelta = actualOutputW.f(i) - targetOutputW.f(i);
			actualOutputDw.putPlus(i, errDelta);
		}
		actualOutputDw.close();
	}
	
	@Override
	public float measure(Matrix actualOutput, Matrix targetOutput) {
		float sum = 0;
		Blob actualOutputW = actualOutput.get("w");
		Blob targetOutputW = targetOutput.get("w");
		for (int i = 0; i < targetOutput.size; i++) {
			float errDelta = actualOutputW.f(i) - targetOutputW.f(i);
			sum += 0.5f * errDelta * errDelta;
		}
		return sum;
	}
}
