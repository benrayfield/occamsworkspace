package immutable.recurrentjava.flop.unary;

import immutable.lazycl.spec.Lazycl;
import immutable.lazycl.spec.LazyclStrictMath;
import immutable.util.Blob;

//TODO always use sigmoid or always use tanh cuz they are just scaled offset versions of eachother.
public class TanhUnit implements Unaflop {

	private static final long serialVersionUID = 1L;

	public float forward(float x){
		return (float)LazyclStrictMath.cpuTanh(x);
	}

	public float deriv(float x) {
		//benrayfield's TODO??? or would it lose precision???...
		//optimize by tanh(x)=sigmoid(2*x)*2-1 aka tanh(x) = (1/(1+e^-(2*x)) * 2 - 1)
		//Sigmoid's derivative just calls exp once and the rest are fast ops,
		//compared to this calling cosh twice.
		double coshx = Math.cosh(x);
		double denom = (Math.cosh(2*x) + 1);
		return (float)(4 * coshx * coshx / (denom * denom));
	}

	public Blob forward(Lazycl lz, Blob floats){
		throw new RuntimeException("TODO");
	}

	public Blob deriv(Lazycl lz, Blob floats){
		throw new RuntimeException("TODO");
	}
}
