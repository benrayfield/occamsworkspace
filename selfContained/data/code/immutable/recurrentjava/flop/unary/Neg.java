package immutable.recurrentjava.flop.unary;

import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;
import mutable.recurrentjava.matrix.Matrix;

/** class by benrayfield */
public class Neg implements Unaflop{

	private static final long serialVersionUID = 1L;

	public float forward(float x){
		return -x;
	}

	public float deriv(float x){
		return -1;
	}
	
	public Blob forward(Lazycl lz, Blob floats){
		return lz.lazycl(
			"Code", "opencl1.2:(global float* out, const float* in){ int id = get_global_id(0); out[id] = -in[id]; }",
			"Bize", floats.bize(),
			"GlobalSize", new int[]{floats.fsizeIntElseThrow()},
			"in", floats
		);
	}
	
	public Blob deriv(Lazycl lz, Blob floats){
		return Matrix.blobOfThisManyNeg1f(floats.fsizeIntElseThrow());
	}
}
