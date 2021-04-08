package immutable.recurrentjava.flop.unary;

import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;
import mutable.recurrentjava.matrix.Matrix;

public class RectifiedLinearUnit implements Unaflop {

	private static final long serialVersionUID = 1L;
	private float slope;
	
	public RectifiedLinearUnit() {
		this.slope = 0;
	}
	
	public RectifiedLinearUnit(float slope) {
		this.slope = slope;
	}
	
	@Override
	public float forward(float x) {
		if (x >= 0) {
			return x;
		}
		else {
			return x * slope;
		}
	}

	@Override
	public float deriv(float x) {
		if (x >= 0) {
			return 1f;
		}
		else {
			return slope;
		}
	}
	
	public Blob forward(Lazycl lz, Blob floats){
		return lz.lazycl(
			"Code", "opencl1.2:(global float* out, const float* in, const float slope){ int id = get_global_id(0); out[id] = (x>=0.0f) ? x : x*slope; }",
			"Bize", floats.bize(),
			"GlobalSize", new int[]{floats.fsizeIntElseThrow()},
			"in", floats,
			"slope", slope
		);
	}
	
	public Blob deriv(Lazycl lz, Blob floats){
		return lz.lazycl(
			"Code", "opencl1.2:(global float* out, const float* in, const float slope){ int id = get_global_id(0); out[id] = (x>=0.0f) ? 1.0f : slope; }",
			"Bize", floats.bize(),
			"GlobalSize", new int[]{floats.fsizeIntElseThrow()},
			"in", floats,
			"slope", slope
		);
	}
}
