package immutable.recurrentjava.flop.unary;

import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;
import mutable.recurrentjava.matrix.Matrix;

public class LinearUnit implements Unaflop {
	

	private static final long serialVersionUID = 1L;

	@Override
	public float forward(float x) {
		return x;
	}

	@Override
	public float deriv(float x) {
		return 1f;
	}
	
	//benrayfield added this
	public static final LinearUnit instance = new LinearUnit();
	
	public Blob forward(Lazycl lz, Blob floats){
		return floats;
	}
	
	public Blob deriv(Lazycl lz, Blob floats){
		return Matrix.blobOfThisMany1f(floats.fsizeIntElseThrow());
	}
}
