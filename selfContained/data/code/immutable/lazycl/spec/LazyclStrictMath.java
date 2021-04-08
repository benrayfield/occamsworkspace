package immutable.lazycl.spec;
import immutable.util.MathUtil;
import immutable.util.Pair;
import mutable.util.Files;

public strictfp class LazyclStrictMath{
	
	/** same bits as gpu computing exp */
	public static double cpuExp(double x){
		return normSubnormals(MathUtil.setLowBitTo0(StrictMath.exp(x)));
		//return StrictMath.exp(x);
		//return MathUtil.setLowBitTo0(StrictMath.exp(x));
	}
	
	public static double cpuSigmoid(double x){
		return 1/(1+cpuExp(-x));
	}
	
	public static double cpuSigmoidDerivative(double x){
		final double sigmoid = cpuSigmoid(x);
		final double oneMinusSigmoid = 1-sigmoid; //force order of ops for determinism
		return sigmoid*oneMinusSigmoid;
	}
	
	public static float cpuSigmoid(float x){
		return (float)cpuSigmoid((double)x);
	}
	
	public static float[] cpuSigmoids(float... x){
		float[] ret = new float[x.length];
		for(int i=0; i<x.length; i++) ret[i] = cpuSigmoid(x[i]);
		return ret;
	}
	
	public static float[] gpuSigmoids(Lazycl lz, float... x){
		return (float[]) lz.opencl().callOpencl(
			Files.readStringFromRelFileCached("/data/lib/fdlibm53/Fdlibm53SigmoidFloatButUsesDoubles.langColonCode"),
			new int[]{x.length}, //globalSize
			null, //localSize
			new float[x.length], //ignores this other than to get its size
			x
		)[0]; //return replacement of x
	}
	
	/** Same as setLowBitTo0(java.lang.StrictMath.exp(x)), computed in opencl,
	though this is wasteful to only do 1 double at a time, its mostly for testing.
	*/
	public static double exp(Lazycl lz, double x){
		return exps(lz, new double[]{x})[0];
	}
	
	/** Same as multiple calls of java.lang.StrictMath.exp(x), computed in opencl in 1 parallel call. */
	public static double[] exps(Lazycl lz, double[] x){
		//TODO optimize by using wrapb (backing double[]) instead of wrapc (copies double[])?
		//Only if caller wont modify the double[] before lazyeval of the returned LazyBlob.
	
		//TODO after callOpenclDependnet works for doubles. As of 2021-2-23 only callOpencl (1 kernel at a time) does.
		boolean useNewCodeForDoubles = false;
		if(useNewCodeForDoubles){
			return exps(lz, lz.wrapc(x)).arr(double[].class);
		}else{
			return (double[]) lz.opencl().callOpencl(
				//readStringFromRelFileCached("/data/lib/fdlibm53/Fdlibm53ExpExceptSetLowBitOfReturnedDoubleTo0.langColonCode"),
					Files.readStringFromRelFileCached("/data/lib/fdlibm53/Fdlibm53Exp.langColonCode"),
				new int[]{x.length},
				null,
				
				//copy output size from this. in callOpencl(Object[]), the param Object[] and returned Object[]
				//are same length, containing all opencl params and returns
				new double[x.length],
				
				x
			)[0];
		}
	}
	
	/** the longs are various things I'm testing as I change the Fdlibm53Exp_withExtraOutputForDebug.langColonCode
	file and Fdlibm53Exp.java together to track down why they're not getting the exact same answer (differs by at most 1 ulp).
	*/
	public static Pair<double[],long[]> exp_withExtraOutputForDebug(Lazycl lz, double[] x){
		Object[] out = lz.opencl().callOpencl(
			Files.readStringFromRelFileCached("/data/lib/fdlibm53/Fdlibm53Exp_withExtraOutputForDebug.langColonCode"),
			new int[]{x.length},
			null,
			
			//copy output size from this. in callOpencl(Object[]), the param Object[] and returned Object[]
			//are same length, containing all opencl params and returns
			new double[x.length],
			new long[x.length],
			x
		);
		return new Pair((double[])out[0], (long[])out[1]);
	}
	
	/** change subnormals to 0. Does not norm infinites or nans. */
	public static double normSubnormals(double d){
		return (-Double.MIN_NORMAL < d && d < Double.MIN_NORMAL) ? 0. : d;
	}
	
	/** Same as multiple calls of java.lang.StrictMath.exp(x), computed in opencl in 1 parallel call.
	For each in double[], same as java.lang.StrictMath.exp(double), returns double[] same size */
	public static LazyBlob exps(Lazycl lz, LazyBlob doubles){
		return lz.lazycl(
			"Code", Files.readStringFromRelFileCached("/data/lib/fdlibm53/Fdlibm53Exp.langColonCode"),
			"Bize", doubles.bize(),
			"GlobalSize", doubles.bize()/64,
			"in", doubles
		);
	}
	
	/** TODO this loses some precision near x=0 by using sigmoid as a middle step
	but the math is exact other than float roundoff.
	TODO test this.
	*/
	public static double cpuTanh(double x){
		final double twoTimesSigmoidOf2X = cpuSigmoid(x*2)*2; //force order of ops for determinism
		return twoTimesSigmoidOf2X-1;
	}
	
	/** TODO test this. */
	public static double cpuTanhDerivative(double x){
		return cpuSigmoidDerivative(x*.5)*.5;
	}

}
