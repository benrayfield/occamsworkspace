package mutable.recurrentjava.loss;
import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.matrix.Matrix;

public class LossArgMax implements Loss {
	
	public final Lazycl lz;
	
	public LossArgMax(Lazycl lz){
		this.lz = lz;
	}

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
		//double maxActual = Double.NEGATIVE_INFINITY;
		//double maxTarget = Double.NEGATIVE_INFINITY;
		float maxActual = Float.NEGATIVE_INFINITY;
		float maxTarget = Float.NEGATIVE_INFINITY;
		int indxMaxActual = -1;
		int indxMaxTarget = -1;
		Blob actualOutputW = actualOutput.get("w");
		Blob targetOutputW = targetOutput.get("w");
		int actualOutputWSize = actualOutputW.fsizeIntElseThrow();
		for (int i = 0; i < actualOutputWSize; i++) {
			if (actualOutputW.f(i) > maxActual){
				maxActual = actualOutputW.f(i);
				indxMaxActual = i;
			}
			if (targetOutputW.f(i) > maxTarget) {
				maxTarget = targetOutputW.f(i);
				indxMaxTarget = i;
			}
		}
		if (indxMaxActual == indxMaxTarget) {
			return 0f;
		}
		else {
			return 1f;
		}
	}
	
}
