/** Ben F Rayfield offers this software opensource MIT license */
package immutable.rbm.learnloop;
import static mutable.util.Lg.*;

import immutable.opencl.OpenCL;
import immutable.util.MathUtil;
import immutable.util.Text;
import mutable.compilers.opencl.lwjgl.LwjglOpenCL;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Time;

/** You can of course use the more general OpenclUtil.callOpencl(String,Object...),
but these are just for convenience of calling some things as normal java funcs.
*/
public class OpenclProgs{
	
	public static OpenCL cl(){
		return LwjglOpenCL.instance();
	}
	
	/** given float[b][c] and float[c][d] returns float[b][d] */
	public static synchronized float[][] matmul(float[][] bc, float[][] cd){
		int bSize = bc.length, cSize = bc[0].length, dSize = cd[0].length;
		if(cd.length != cSize) throw new Error("Sizes dont match");
		//FIXME verify sizes match and are rectangle arrays
		float[] bd1d = matmul(bSize, cSize, dSize, MathUtil.array2dTo1d(bc), MathUtil.array2dTo1d(cd));
		return MathUtil.array1dTo2d(bd1d,bSize);
	}
	
	/** given double[b][c] and double[c][d] returns double[b][d] */
	public static synchronized double[][] matmul(double[][] bc, double[][] cd){
		int bSize = bc.length, cSize = bc[0].length, dSize = cd[0].length;
		if(cd.length != cSize) throw new Error("Sizes dont match");
		//FIXME verify sizes match and are rectangle arrays
		double[] bd1d = matmul(bSize, cSize, dSize, MathUtil.array2dTo1d(bc), MathUtil.array2dTo1d(cd));
		return MathUtil.array1dTo2d(bd1d,bSize);
	}
	
	static boolean isFirstCallOf_matmulThenSigmoid = true;
	
	/** lower lag than doing multiple opencl calls using matmul first */
	public static synchronized float[][] matmulWithBiasThenSigmoid(float[][] bias, float[][] bc, float[][] cd){
		if(isFirstCallOf_matmulThenSigmoid){ //FIXME testing a theory that opencl needs to run it once which has nans and later it works?
			isFirstCallOf_matmulThenSigmoid = false;
			matmulWithBiasThenSigmoid(bias,bc,cd);
		}
		int bSize = bc.length, cSize = bc[0].length, dSize = cd[0].length;
		if(cd.length != cSize) throw new Error("Sizes dont match");
		if(bias.length != bc.length || bias[0].length != cd[0].length) {
			throw new Error("Sizes dont match");
		}
		//FIXME verify sizes match and are rectangle arrays
		float[] bd1d = matmulWithBiasThenSigmoid(MathUtil.array2dTo1d(bc), bSize, cSize, dSize, MathUtil.array2dTo1d(bc), MathUtil.array2dTo1d(cd));
		return MathUtil.array1dTo2d(bd1d,bSize);
	}
	
	public static synchronized float[][] matmulWithBiasThenSigmoid_usingCpu(float[] bias, float[][] bc, float[][] cd){
		/*"opencl1.2:(global const float* bias, int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){\r\n"+
				"	int bd = get_global_id(0);\r\n"+
				"	const int b = bd/dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?//
				"	const int d = bd%dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?
				"	float sum = bias[bd];\r\n"+
				"	for(int c=0; c<cSize; c++){\r\n"+
				"		sum += bc[b*cSize+c]*cd[c*dSize+d];\r\n"+ //TODO optimize allow get_global_id(more dims)?
				"	}\r\n"+
				"	float chance = 1/(1+exp(-sum));\r\n"+
				"	bdOut[bd] = chance;\r\n"+
				"}";
		*/
		throw new Error("TODO");
	}
	
	/** lower lag than doing multiple opencl calls using matmul first */
	public static synchronized float[][] matmulThenSigmoidThenWeightedCoinFlip(float[][] bias, float[][] bc, float[][] cd){
		int bSize = bc.length, cSize = bc[0].length, dSize = cd[0].length;
		if(cd.length != cSize) throw new Error("Sizes dont match");
		if(bias.length != bc.length || bias[0].length != cd[0].length) {
			throw new Error("Sizes dont match");
		}
		//FIXME verify sizes match and are rectangle arrays
		float[] bd1d = matmulThenSigmoidThenWeightedCoinFlip(MathUtil.array2dTo1d(bias), bSize, cSize, dSize, MathUtil.array2dTo1d(bc), MathUtil.array2dTo1d(cd));
		return MathUtil.array1dTo2d(bd1d,bSize);
	}
	
	/** bc.length==bSize*cSize && cd.length==cSize*dSize */
	public static synchronized float[] matmul(int bSize, int cSize, int dSize, float[] bc, float[] cd){
		Object[] out = cl().callOpencl(
			
			//FIXME slower, try this until get the right answer then start using matmulCode1dAs2d instead and make that work
			matmulCode1dAs2d, new int[]{bSize*dSize},
			null,
			
			//FIXME This gets about 3.5 gflops on my 4x1.6GhzLaptop, while the other only about 2. Both give wrong answer,
			//this one gives 0 and other one gives it appears 1 of the input numbers, so I'm going back to the slower 1d one
			//while I fix that then come back to this for speed if I can
			//matmulCode2d, new int[]{bSize, dSize},
			
			bSize, cSize, dSize, bc, cd, new float[bSize*dSize]);
		return (float[]) out[out.length-1];
	}
	
	/** bc.length==bSize*cSize && cd.length==cSize*dSize
	WARNING THIS HASNT BEEN TESTED.
	*/
	public static synchronized double[] matmul(int bSize, int cSize, int dSize, double[] bc, double[] cd){
		Object[] out = cl().callOpencl(
			openclNdrangeCode_matmulDouble, new int[]{bSize*dSize}, null,
			bSize, cSize, dSize, bc, cd, new double[bSize*dSize]);
		return (double[]) out[out.length-1];
	}
	
	
	
	/** bc.length==bSize*cSize && cd.length==cSize*dSize */
	public static synchronized float[] matmulWithBiasThenSigmoid(float[] bias, int bSize, int cSize, int dSize, float[] bc, float[] cd){
		Object[] out = cl().callOpencl(matmulCode1dAs2dThenSigmoid, new int[]{bSize*dSize}, null,
			bias, bSize, cSize, dSize, bc, cd, new float[bSize*dSize]);
		return (float[]) out[out.length-1];
	}
	
	/** bc.length==bSize*cSize && cd.length==cSize*dSize */
	public static synchronized float[] matmulThenSigmoidThenWeightedCoinFlip(float[] bias, int bSize, int cSize, int dSize, float[] bc, float[] cd){
		/*
		//FIXME remove this test
		Object[] outTest = OpenclUtil.callOpencl("opencl1.2:(global float* bdOut){\r\n"+
				"	int bd = get_global_id(0);\r\n"+
				"	float x = bd;\r\n"+
				"	x = x*x;\r\n"+
				"	bdOut[bd] = x;\r\n"+
				"}", new int[]{100},
				new float[100]);
		float[] outF = (float[])outTest[0];
		System.out.println("FIXME remove this test. outF[57]="+outF[57]);
		*/
		
		Object[] out = cl().callOpencl(matmulCode1dAs2dThenSigmoidThenWeightedCoinFlip, new int[]{bSize*dSize}, null,
			bias, bSize, cSize, dSize, bc, cd, new float[bSize*dSize]);
		return (float[]) out[out.length-1];
	}
	
	/**
	https://www.reddit.com/r/gpgpu/comments/bklzru/my_float_code_works_but_double_code_throws_how/
	TODO
	https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/scalarDataTypes.html
	QUOTE
		Optional Double Precision and Half Floating Point
		OpenCL 1.0 adds support for double precision and half floating-point as optional extensions.
		The double data type must confirm to the IEEE-754 double precision storage format.
		
		An application that wants to use double will need to include the
			#pragma OPENCL EXTENSION cl_khr_fp64 : enable
			https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/cl_khr_fp64.html
		directive before any double precision data type is declared in the kernel code. This will extended the list of built-in vector and scalar data types to include the following:
		
		Type in OpenCL Language	Description	API type for application
		double	A double precision float.	cl_double
		double2	A 2-component double vector.	cl_double2
		double4	A 4-component double vector.	cl_double4
		double8	An 8-component double vector.	cl_double8
		double16	A 16-component double vector.	cl_double16
	UNQUOTE.
	*/
	public static final String openclNdrangeCode_matmulDouble =
		"opencl1.2:(int const bSize, int const cSize, int const dSize, global const double* bc, global const double* cd, global double* bdOut){\r\n"+
		"	int bd = get_global_id(0);\r\n"+
		"		const int b = bd/dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?//
		"		const int d = bd%dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"		double sum = 0;\r\n"+
		"		for(int c=0; c<cSize; c++){\r\n"+
		"			sum += bc[b*cSize+c]*cd[c*dSize+d];\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"		}\r\n"+
		"		bdOut[bd] = sum;\r\n"+
		"}";
	
	public static final String matmulCode1dAs2d =
		"opencl1.2:(int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){\r\n"+
		"	int bd = get_global_id(0);\r\n"+
		"		const int b = bd/dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?//
		"		const int d = bd%dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"		float sum = 0;\r\n"+
		"		for(int c=0; c<cSize; c++){\r\n"+
		"			sum += bc[b*cSize+c]*cd[c*dSize+d];\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"		}\r\n"+
		"		bdOut[bd] = sum;\r\n"+
		"}";
	
	public static final String matmulCode1dAs2dThenSigmoid =
		"opencl1.2:(global const float* bias, int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){\r\n"+
		"	int bd = get_global_id(0);\r\n"+
		"	const int b = bd/dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?//
		"	const int d = bd%dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"	float sum = bias[bd];\r\n"+
		"	for(int c=0; c<cSize; c++){\r\n"+
		"		sum += bc[b*cSize+c]*cd[c*dSize+d];\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"	}\r\n"+
		"	float chance = 1/(1+exp(-sum));\r\n"+
		"	bdOut[bd] = chance;\r\n"+
		"}";
	
	public static final String matmulCode1dAs2dThenSigmoidThenWeightedCoinFlip =
		"opencl1.2:(global const float* bias, int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){\r\n"+
		"	int bd = get_global_id(0);\r\n"+
		"	int b = bd/dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?//
		"	int d = bd%dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"	float sum = bias[bd];\r\n"+
		"	for(int c=0; c<cSize; c++){\r\n"+
		"		sum += bc[b*cSize+c]*cd[c*dSize+d];\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"	}\r\n"+
		"	float zero = 0.0f;\r\n"+
		"	float one = 1.0f;\r\n"+
		"	float fourNineNineNineNine = 49999.0f;\r\n"+
		"	float chance = one/(one+exp(-sum));\r\n"+
		"	float randFraction = fmod(fabs(sum)*fourNineNineNineNine,one);\r\n"+
		"	float weightedCoinFlip = fmax(zero,ceil(chance-randFraction));\r\n"+
		"	bdOut[bd] = weightedCoinFlip;\r\n"+
		//"	bdOut[bd] = one+bias[bd]*3.1f;\r\n"+
		"}";
	
	/*public static final String matmulCode1dAs2d =
		"kernel void "+OpenclUtil.newKernelName()+"(int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){\r\n"+
		"	const int bd = get_global_id(0);\r\n"+
		"	const int b = bd/dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"	const int d = bd%dSize;\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"	int bcOffset = b*cSize;\r\n"+
		"	int cdOffset = d;\r\n"+
		//"	if(itemId < size){\r\n"+ //FIXME why did Lwjgl sample code do this? They have indexing problems?
		"	float sum = 0;\r\n"+
		"	for(int c=0; c<cSize; c++){\r\n"+
		"		sum += bc[bcOffset]*cd[cdOffset];\r\n"+
		"		bcOffset++;\r\n"+
		"		cdOffset += dSize;\r\n"+
		//"		sum += bc[b*cSize+c]*cd[c*dSize+d];\r\n"+ //TODO optimize allow get_global_id(more dims)?
		"	}\r\n"+
		"	bdOut[bd] = sum;\r\n"+
		//"		result[itemId] = a[itemId] + b[itemId];\r\n"+
		//"	}\r\n"+
		"}";
	*/
	
	/*public static final String matmulCode2d =
		//	FIXME verify the indexs are right
		"kernel void "+OpenclUtil.newKernelName()+"(int const bSize, int const cSize, int const dSize, global const float* bc, global const float* cd, global float* bdOut){\r\n"+
		"	int b = get_global_id(0);\r\n"+
		"	int d = get_global_id(1);\r\n"+
		//"	int d = b;\r\n"+ //FIXME
		//"	float sum = 0;\r\n"+
		//"	int bcOffset = b*cSize;\r\n"+
		//"	int cdOffset = d;\r\n"+
		//"	for(int c=0; c<cSize; c++){\r\n"+
		//"		sum += bc[bcOffset]*cd[cdOffset];\r\n"+
		//"		bcOffset++;\r\n"+
		//"		cdOffset += dSize;\r\n"+
		//"	}\r\n"+
		"	bdOut[b*dSize+d] = 1;\r\n"+
		"}";*/
	
	public static float[][] matmulCpu(float[][] bc, float[][] cd){
		int B = bc.length;
		int C = bc[0].length;
		int D = cd[0].length;
		//FIXME verify sizes match and are rectangle arrays
		float[][] bd = new float[B][D];
		for(int b=0; b<B; b++){
			for(int d=0; d<D; d++){
				float sum = 0;
				for(int c=0; c<C; c++){
					sum += bc[b][c]*cd[c][d];
				}
				bd[b][d] = sum;
			}
		}
		return bd;
	}
	
	public static double[][] matmulCpu(double[][] bc, double[][] cd){
		int B = bc.length;
		int C = bc[0].length;
		int D = cd[0].length;
		//FIXME verify sizes match and are rectangle arrays
		double[][] bd = new double[B][D];
		for(int b=0; b<B; b++){
			for(int d=0; d<D; d++){
				double sum = 0;
				for(int c=0; c<C; c++){
					sum += bc[b][c]*cd[c][d];
				}
				bd[b][d] = sum;
			}
		}
		return bd;
	}
	
	public static void testOpencl(boolean allowDoubles){
		//testInt();
		testOpencl_matmulFloat();
		testOpencl_matmulFloat();
		if(allowDoubles){
			testOpencl_matmulDouble();
			testOpencl_matmulDouble();
		}
	}
	
	public static void testOpencl_matmulFloat(){
		lg("Testing with random arrays...");
		int bSize = 50, cSize = 30, dSize = 70;
		float[][] bc = new float[bSize][cSize];
		float[][] cd = new float[cSize][dSize];
		for(int c=0; c<cSize; c++){
			for(int b=0; b<bSize; b++){
				bc[b][c] = (float)Rand.strongRand.nextGaussian();
			}
			for(int d=0; d<dSize; d++){
				cd[c][d] = (float)Rand.strongRand.nextGaussian();
			}
		}
		float[][] bdFromCpu = matmulCpu(bc, cd);
		float[][] bdFromOpencl = matmul(bc, cd);
		double sumOfSquares = 0;
		double sumOfSquaresOfCpu = 0, sumOfSquaresOfOpencl = 0;
		boolean isExact = true;
		for(int b=0; b<bSize; b++){
			for(int d=0; d<dSize; d++){
				float sub = bdFromCpu[b][d]-bdFromOpencl[b][d];
				if(sub != 0) isExact = false;
				sumOfSquares += sub*sub;
				//Cuz opencl got the right answer but stdDevOfErr=0.0
				//WARNING: An illegal reflective access operation has occurred
				//WARNING: Illegal reflective access by org.lwjgl.LWJGLUtil$3 (file:/C:/q29x/eclw/3/HumanAiNet_2019-2+_todoClibin/src/data/lib/lwjgl-debug.jar) to method java.lang.ClassLoader.findLibrary(java.lang.String)
				//WARNING: Please consider reporting this to the maintainers of org.lwjgl.LWJGLUtil$3
				//WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
				//WARNING: All illegal access operations will be denied in a future release
				//testOpencl matmul passed, stdDevOfErr=0.0
				sumOfSquaresOfCpu += bdFromCpu[b][d]*bdFromCpu[b][d];
				sumOfSquaresOfOpencl += bdFromOpencl[b][d]*bdFromOpencl[b][d];
			}
		}
		int samples = bSize*dSize;
		double stdDevOfErr = Math.sqrt(sumOfSquares/samples);
		String result = "stdDevOfErr="+stdDevOfErr+" sumOfSquaresOfCpu="+sumOfSquaresOfCpu+" sumOfSquaresOfOpencl="+sumOfSquaresOfOpencl;
		if(stdDevOfErr > .000001) throw new Error("matmul differs too much between cpu and opencl, "+result);
		lg("testOpencl_matmulFloat matmul passed !strictfp, "+result);
		if(!isExact) throw new Error("testOpencl_matmulFloat failed strictfp");
		lg("testOpencl_matmulFloat matmul passed strictfp");
	}
	
	public static void testOpencl_matmulDouble(){
		lg("Testing with random arrays...");
		int bSize = 50, cSize = 30, dSize = 70;
		double[][] bc = new double[bSize][cSize];
		double[][] cd = new double[cSize][dSize];
		for(int c=0; c<cSize; c++){
			for(int b=0; b<bSize; b++){
				bc[b][c] = Rand.strongRand.nextGaussian();
			}
			for(int d=0; d<dSize; d++){
				cd[c][d] = Rand.strongRand.nextGaussian();
			}
		}
		double[][] bdFromCpu = matmulCpu(bc, cd);
		double[][] bdFromOpencl = matmul(bc, cd);
		double sumOfSquares = 0;
		double sumOfSquaresOfCpu = 0, sumOfSquaresOfOpencl = 0;
		for(int b=0; b<bSize; b++){
			for(int d=0; d<dSize; d++){
				double sub = bdFromCpu[b][d]-bdFromOpencl[b][d];
				sumOfSquares += sub*sub;
				//Cuz opencl got the right answer but stdDevOfErr=0.0
				//WARNING: An illegal reflective access operation has occurred
				//WARNING: Illegal reflective access by org.lwjgl.LWJGLUtil$3 (file:/C:/q29x/eclw/3/HumanAiNet_2019-2+_todoClibin/src/data/lib/lwjgl-debug.jar) to method java.lang.ClassLoader.findLibrary(java.lang.String)
				//WARNING: Please consider reporting this to the maintainers of org.lwjgl.LWJGLUtil$3
				//WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
				//WARNING: All illegal access operations will be denied in a future release
				//testOpencl matmul passed, stdDevOfErr=0.0
				sumOfSquaresOfCpu += bdFromCpu[b][d]*bdFromCpu[b][d];
				sumOfSquaresOfOpencl += bdFromOpencl[b][d]*bdFromOpencl[b][d];
			}
		}
		int samples = bSize*dSize;
		double stdDevOfErr = Math.sqrt(sumOfSquares/samples);
		String result = "stdDevOfErr="+stdDevOfErr+" sumOfSquaresOfCpu="+sumOfSquaresOfCpu+" sumOfSquaresOfOpencl="+sumOfSquaresOfOpencl;
		if(stdDevOfErr > .000001) throw new Error("matmul differs too much between cpu and opencl, "+result);
		lg("testOpencl_matmulDouble matmul passed, "+result);
	}
	
	/*public static final String simpleTest =
		"kernel void "+OpenclUtil.newKernelName()+"(global const float* in, global float* out, int const size){\r\n"+
		"	const int i = get_global_id(0);\r\n"+
		//"	if(0 <= i && i < size){\r\n"+
		"		const float b = in[i];\r\n"+
		//"		out[i] = (b*4);\r\n"+
		//"		float r = (float)size;\r\n"+
		"		const float r = (b*2);\r\n"+
		"		out[i] = r;\r\n"+
		//"	}\r\n"+
		"}";
	
	public static void main(String[] args){
		int size = 20;
		float[] a = new float[size], b = new float[size];
		for(int i=0; i<size; i++){
			a[i] = 1000+i;
			b[i] = 2000+i;
		}
		Object[] outs = OpenclUtil.callOpencl(simpleTest, new int[]{size}, a, b, size);
		float[] out = (float[])outs[1];
		for(int i=0; i<size; i++){
			System.out.println("out["+i+"]="+out[i]+" (intbits)"+Float.floatToIntBits(out[i]));
		}
	}*/
	
	
	/** matmul is probably bottlenecked by IO moving between globalMem and gpuCores
	as it seems to be many times slower than the theoretical max flops of the gpu,
	so I'm testing this which wont have any global memory at all except to write 1 number
	per millions or more of calculations that dont read from global memory
	and instead are derived from get_global_id(0).
	*/
	public static void testGpuComputeFlopsWithoutMuchGlobalMem(){
		int threads =  1000000;
		int loopSize = 1000000;
		int opsPerLoopBody = 5; //or 4 if z = x * x and its z*z*x instead of x*x*x*x*x
		double totalOps = (double)threads*loopSize*opsPerLoopBody;
		String code =
			"kernel void testGpuComputeFlopsWithoutMuchGlobalMem(int const a, global float* theOut){\n"+
			"	int id = get_global_id(0);\n"+
			"	float sum = 0;\n"+
			"	for(int x=id; x<a+id; x++){\n"+
			//"		sum += x;\n"+
			"		sum += x*x*x*x*x;\n"+
			//"		sum += x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x*x;\n"+
			"	}\n"+
			"	theOut[id] = sum;\n"+
			"}";
		Object[] outs = cl().callOpencl( //compile and run it once before timing it
			code,
			new int[]{100},
			null,
			120,
			new float[100] //ignored
		);
		double timeStart = Time.now();
		outs = cl().callOpencl(
			code,
			new int[]{threads},
			null,
			loopSize,
			new float[threads] //ignored
		);
		float[] out = (float[]) outs[1];
		double timeEnd = Time.now();
		double duration = timeEnd-timeStart;
		double flops = totalOps/duration;
		double gflops = flops*1e-9;
		lg("outs[0] = "+out[0]);
		lg("outs[5] = "+out[5]);
		lg("gflops="+gflops+" seconds="+duration+" ops="+totalOps);
	}
	
	public static void main(String... args){
		testOpencl(false);
		testGpuComputeFlopsWithoutMuchGlobalMem();
	}
	

}
