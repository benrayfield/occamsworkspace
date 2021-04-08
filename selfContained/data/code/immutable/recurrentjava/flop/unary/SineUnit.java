package immutable.recurrentjava.flop.unary;

import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;

public class SineUnit implements Unaflop{

	private static final long serialVersionUID = 1L;

	public float forward(float x) {
		return (float)StrictMath.sin(x);
	}

	public float deriv(float x) {
		return (float)StrictMath.cos(x);
	}
	
	public Blob forward(Lazycl lz, Blob floats){
		throw new RuntimeException("TODO make sure its deterministic and computes same bits in cpu and gpu");
	}
	
	public Blob deriv(Lazycl lz, Blob floats){
		throw new RuntimeException("TODO make sure its deterministic and computes same bits in cpu and gpu");
	}
}
