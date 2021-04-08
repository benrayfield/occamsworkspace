package mutable.recurrentjava.loss;
import immutable.util.Blob;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.matrix.Matrix;

public class LossMultiDimensionalBinary implements Loss {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public void backward(Matrix actualOutput, Matrix targetOutput) {
		throw new UnsupportedOperationException();
	}
	
	@Override
	public float measure(Matrix actualOutput, Matrix targetOutput) {
		if (actualOutput.size != targetOutput.size) {
			throw new Error("mismatch");
		}
		
		Blob actualOutputW = actualOutput.get("w");
		Blob targetOutputW = targetOutput.get("w");
		
		for (int i = 0; i < targetOutput.size; i++) {
			if (targetOutputW.f(i) >= 0.5 && actualOutputW.f(i) < 0.5) {
				return 1;
			}
			if (targetOutputW.f(i) < 0.5 && actualOutputW.f(i) >= 0.5) {
				return 1;
			}
		}
		return 0;
	}

}
