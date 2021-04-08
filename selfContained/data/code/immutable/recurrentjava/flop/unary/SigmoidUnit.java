package immutable.recurrentjava.flop.unary;

import data.lib.fdlibm53.Fdlibm53Exp;
import immutable.lazycl.spec.Lazycl;
import immutable.lazycl.spec.LazyclStrictMath;
import immutable.util.Blob;
import mutable.util.Files;

public class SigmoidUnit implements Unaflop {

	private static final long serialVersionUID = 1L;

	@Override
	public float forward(float x) {
		return (float)(1 / (1 + LazyclStrictMath.cpuExp(-x)));
	}

	//benrayfield renamed this to deriv
	@Override
	public float deriv(float x) {
		float act = forward(x);
		return act * (1 - act);
	}
	
	//benrayfield added this
	public static final SigmoidUnit instance = new SigmoidUnit();
	
	public Blob forward(Lazycl lz, Blob floats){
		return lz.lazycl(
			"Code", Files.readStringFromRelFileCached("/data/lib/fdlibm53/Fdlibm53SigmoidFloatButUsesDoubles.langColonCode"),
			"Bize", floats.bize(),
			"GlobalSize", new int[]{floats.fsizeIntElseThrow()},
			"in", floats
		);
	}
	
	public Blob deriv(Lazycl lz, Blob floats){
		return lz.lazycl(
			"Code", Files.readStringFromRelFileCached("/data/lib/fdlibm53/Fdlibm53DerivativeOfSigmoidFloatButUsesDoubles.langColonCode"),
			"Bize", floats.bize(),
			"GlobalSize", new int[]{floats.fsizeIntElseThrow()},
			"in", floats
		);
	}
}
