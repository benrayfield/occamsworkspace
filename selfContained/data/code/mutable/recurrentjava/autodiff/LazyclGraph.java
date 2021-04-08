package mutable.recurrentjava.autodiff;

import java.util.ArrayList;
import java.util.List;
import java.util.NavigableSet;
import java.util.TreeSet;
import java.util.function.Consumer;

import org.lwjgl.PointerBuffer;

import immutable.acyclicflow.AcyclicFlowF;
import immutable.lazycl.spec.Lazycl;
import immutable.rbm.learnloop.OpenclProgs;
import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.rnn.RnnParams;
import immutable.util.Blob;
import immutable.util.MathUtil;
import mutable.compilers.opencl.lwjgl.Lwjgl;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.matrix.MatrixCache;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.trainer.Trainer;
import mutable.util.task.RunnableTask;
import mutable.util.task.Task;

/** a low lag (< .01 second for 10 sequential opencl ndrange kernels, but varies by GPU and params etc,
but TODO its only < .01 second for 1 ndrange kernel until the upgrade of vm_evalOneTheSlowWayNotInGroups(LazyBlob))
Graph optimized for GPU (using lazycl which by default 2021-4-6 uses lwjgl2 opencl1.2,
which I've tested on win10 (and an earlier version on win7 and linux on an AMD gpu,
and should work many places if you replace lwjgl64.dll with the file for your OS,
and if that doesnt work you can implement the immutable.opencl.OpenCL interface and use that in Lazycl constructor).
A Graph is a mutable builder of numberCrunching ops with backprop (and in some cases also training) built in.
*/
public strictfp class LazyclGraph implements Graph{
	
	//FIXME this code was 2021-4-6 copied from CpuGraph but needs to be rewritten to
	//create LazyBlobs by Lazycl.lazycl(...)
	//instead of cpu computed loops of things like "m1dw.putPlus(i, neuron.deriv(m1w.f(i)) * outDw.f(i));".
	//For number crunching related things,
	//even things which dont benefit from GPU (vs CPU), still compute them in GPU for the low lag
	//of not having to copy back to CPU in the middle of a sequence of LazyBlob calculations.
	
	//TODO create acyclicFlow functions here that can do multiple steps within Gru nodes and Lstm nodes for lower lag,
	//instead of creating a Matrix for each multiply, add, etc. Forward would happen in 1 opencl ndrange kernel,
	//and backprop would happen in 1, instead of as many as number of Matrixs. This will make it practical
	//to learn in realtime, not at gaming-low-lag but maybe a lag of 0.2 second for learning the next batch
	//since it will be a few hundred ndrange kernels deep (unroll backprop of time steps of Gru or Lstm neuralnet),
	//and the prediction will have a lag of .01 second which is gaming-low-lag.
	//Also, using CpuGraph, small amounts of learning can be done in .01 second,
	//if you use a LazyclGraph for efficient GPU learning (.2 seconds lag) and CpuGraph for what happens during those .2 seconds
	//so user would experience instant learning, and sum the 2 changes of weights together.
	//So the whole thing can, in theory, be done at gaming-low-lag.
	
	public final Lazycl lz;
	
	protected boolean applyBackprop;
	public boolean isApplyBackprop(){ return applyBackprop; }
	
	/** tasks to do before backprop, if any.
	benrayfield added this to put DependnetOps in (UPDATE will use LazyBlob which internally uses DependOp),
	the parts that normally happen as soon as a Matrix is created will instead
	be lazyEvaled all at once in opencl, or when opencl is not used then still instant.
	*/
	public final List<Task> forwardprop = new ArrayList<>();
	
	public final List<Task> backpropToDoInReverseOrder = new ArrayList<>();
	
	/** tasks to do for training after forwardprop and backprop, if any */
	public final List<Task> trainprop = new ArrayList<>();
	
	public LazyclGraph(Lazycl lz){
		this(lz,true);
	}
	
	public LazyclGraph(Lazycl lz, boolean applyBackprop){
		this.lz = lz;
		this.applyBackprop = applyBackprop;
	}
	
	public void learn(){
		List<Task> tasks = new ArrayList(forwardprop);
		tasks.addAll(MathUtil.reverse(backpropToDoInReverseOrder));
		tasks.addAll(trainprop);
		if(tasks.stream().allMatch(x->(x instanceof RunnableTask))){
			//CPU creates LazyBlobs which, when their contents are observed (Blob.f(int) etc) later,
			//are computed in GPU, or (TODO) some or all parts of that may be computed in GPU now depending
			//if there is enough memory to delay the computing. As of 2021-4-6 lazycl just waits for
			//the first observe in any LazyBlob then computes in GPU the contents of all LazyBlobs not computed yet,
			//one opencl ndrange kernel at a time (LazyclPrototype.vm_evalOneTheSlowWayNotInGroups(LazyBlob)).
			//The "groups" way is a TODO redesign it for lower lag, to do multiple ndrange kernels
			//in GPU before copying anything back to memory that can be read by CPU.
			//It can change between CPU and GPU about 100 times per second,
			//so if you do 15 ndrange kernels per call of GPU, one after the other,
			//then it can do 1500 ndrange kernels per second. It cant do much more than that per second,
			//even if each ndrange kernel only changes 1 byte and does no significant work.
			//You lose efficiency if its divided into many smaller pieces of work.
			//It seems to reach max efficiency of matmul (matrix multiply) when its at least float[1024][1024]
			//which is around 2 billion flops, which it can do 60 times per second on a new GPU as of 2021.
			//It can do far more flops when theres less copying between GPU cores. I've seen it do ~2 teraflops.
			
			Task.doTasksInCpu(tasks);
			
		}else{
			throw new Error("cant Task.doTasksInOpencl(tasks); cuz redesigning to use CpuGraph and OpenclGraph");
		}
		forwardprop.clear();
		backpropToDoInReverseOrder.clear();
		trainprop.clear();
	}
	
	/** (benrayfield made this func) Was it wrong to copy the stepcache?
	I dont see other funcs accessing stepcache places other than in Trainer.
	Maybe thats cuz this op isnt used except inside neural nodes
	and only the Model.getParameters() (such as weights) use stepCache.
	*/
	public Matrix concatVectors(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		//FIXME backprop cant branch 2 paths forward from concat cuz
		//it will copy everything backward including "w". Normally backprop doesnt copy "w",
		//but this isnt really a backprop, just a translation out=cat(m1,m2) both directions.
		final Matrix out = new Matrix(lz, m1.size+m2.size);
		NavigableSet<String> keys = new TreeSet(m1.keys());
		keys.addAll(m2.keys());
		//String key = "w";
		for(String key : keys){
			out.put(key, catBlock32(lz, m1.get(key), m2.get(key)));
		}
		if(applyBackprop){
			m1.disable_canBeParamInGraphAgain();
			m2.disable_canBeParamInGraphAgain();
			back(()->{
				for(String key : keys){
					//FIXME??? what if new keys are added to the concatted Matrix?
					//Probably wouldnt need them earlier since they werent created before the concat.
					Blob outBlob = out.get(key);
					m1.put(key, rangeBlock32(lz, outBlob, 0, m1.size));
					m2.put(key, rangeBlock32(lz, outBlob, m1.size, m1.size+m2.size));
				}
			});
		}
		return out;
	}
	
	/** concat if they are in blocks of 32, using GPU, lazy of course. The nonlazy part is done instantly in CPU.
	This could probably be done faster in CPU except for switching between GPU and CPU costs .003 milliseconds.
	<br><br>
	FIXME as of 2021-4-8  I havent done much testing of floats and ints mixing
	in CLMem params of ndrange kernel, so if theres some cant convert float/int error look here.
	I actually plan to use this for floats, but ints are more general.
	*/
	public static Blob catBlock32(Lazycl lz, Blob a, Blob b){
		if((a.bize()%32) != 0 || b.bize()%32 != 0) throw new RuntimeException(
			"Sizes must be multiple of 32 bits: "+a.bize()+" "+b.bize());
		long retSize = a.bize()+b.bize();
		if(retSize < 0 || retSize > Integer.MAX_VALUE*32L) throw new RuntimeException(
			"Too big: "+a.bize()+" "+b.bize());
		return lz.lazycl(
			"Code",
				"opencl1.2:(global float* out, const global int* a, const int aSize const global int* b){"+n+
				"	int id = get_global_id(0);"+n+
				"	out[id] = (id<aSize) ? a[id] : b[id-aSize];"+n+
				"}",
			"Bize", retSize,
			"GlobalSize", (int)(retSize/32), //int[1] and int are same as Bize. TODO change the other int[]{...} if [1].
			"a", a,
			"aSize", a.fsizeIntElseThrow(), //size in units of floats or ints (both 32 bits)
			"b", b
		);
	}
	
	/** start and endExcl are in units of ints/floats.
	<br><br>
	FIXME as of 2021-4-8  I havent done much testing of floats and ints mixing
	in CLMem params of ndrange kernel, so if theres some cant convert float/int error look here.
	I actually plan to use this for floats, but ints are more general.
	*/
	public static Blob rangeBlock32(Lazycl lz, Blob a, int start, int endExcl){
		if(a.bize()%32 != 0 || a.bize() > Integer.MAX_VALUE*32L) throw new RuntimeException(
			"Size must be multiple of 32 bits: "+a.bize());
		if(start < 0 || endExcl < start || a.bize()/32 < endExcl) throw new RuntimeException(
			"start="+start+" endExcl="+endExcl+" blob size in ints/floats = "+(a.bize()/32));
		return lz.lazycl(
			"Code",
				"opencl1.2:(global float* out, const global int* in, const int offset){"+n+
				"	out[id] = in[offset+get_global_id(0)];"+n+
				"}",
			"Bize", (endExcl-start)*32L,
			"GlobalSize", endExcl-start,
			"in", a,
			"offset", start
		);
	}
	 
	protected void back(Runnable r){
		backpropToDoInReverseOrder.add(new RunnableTask(r));
	}
		//m1dw.putPlus(i, neuron.deriv(m1w.f(i)) * outDw.f(i));
	
	public Matrix nonlin(final Unaflop neuron, final Matrix in){
		Matrix.throwUnless_canBeParamInGraphAgain(in);
		final Matrix out = new Matrix(lz, in.rows, in.cols);
		out.put("w", neuron.forward(lz, in.get("w")));
		if(applyBackprop) back(()->{
			//m1dw.putPlus(i, neuron.deriv(m1w.f(i)) * outDw.f(i)); renaming to...
			//inDw.putPlus(i, neuron.deriv(inW.f(i)) * outDw.f(i));
			in.put("dw", neuron.x_plus_derivOf_yTimesZ(lz, in.get("dw"), in.get("w"), out.get("dw")));
		});
		return out;
	}
	
	public static final String n = "\n";
	
	/** nvidia warp size is 32, so this is efficient to be multiple of that,
	but should also work on AMD gpus etc. TODO verify.
	*/
	public static final int TS = 32;
	
	public static final int WPT = TS;
	public static final int RTS = TS/WPT;
	
	/** slightly modified myGEMM2 from tutorial: https://cnugteren.github.io/tutorial/pages/page4.html
	opensource MIT license: https://github.com/CNugteren/myGEMM/blob/master/LICENSE
	https://stackoverflow.com/questions/51003876/opencl-global-vs-global-and-kernel-vs-kernel says __global and global are same keyword.
	See mutable.compilers.opencl.TestOpenclLocalMem.
	*/
	public static final String matmulFCode = "opencl1.2:(__global float* xz, const __global float* xy, const __global float* yz, const int x, const int y, const int z){"+n+
		""+n+
		"// Thread identifiers"+n+
		"const int row = get_local_id(0); // Local row ID (max: TS)"+n+
		"const int col = get_local_id(1); // Local col ID (max: TS)"+n+
		"const int globalRow = "+TS+"*get_group_id(0) + row; // Row ID of c (0..m)"+n+
		"const int globalCol = "+TS+"*get_group_id(1) + col; // Col ID of c (0..N)"+n+
		""+n+
		"// Local memory to fit a tile of TS*TS elements of xy and yz"+n+
		"__local float Asub["+TS+"]["+TS+"];"+n+
		"__local float Bsub["+TS+"]["+TS+"];"+n+
		""+n+
		"// Initialise the accumulation register"+n+
		"float acc = 0.0f;"+n+
		""+n+
		"// Loop over all tiles"+n+
		"const int numTiles = y/"+TS+";"+n+
		"for (int t=0; t<numTiles; t++) {"+n+
		"	// Load one tile of xy and yz into local memory"+n+
		"	const int tiledRow = "+TS+"*t + row;"+n+
		"	const int tiledCol = "+TS+"*t + col;"+n+
		"	Asub[col][row] = xy[tiledCol*x + globalRow];"+n+
		"	Bsub[col][row] = yz[globalCol*y + tiledRow];"+n+
		"	"+n+
		"	// Synchronise to make sure the tile is loaded"+n+
		"	barrier(CLK_LOCAL_MEM_FENCE);"+n+
		"	"+n+
		"	// Perform the computation for a single tile"+n+
		"	for (int ki=0; ki<"+TS+"; ki++) {"+n+
		"		acc += Asub[ki][row] * Bsub[col][ki];"+n+
		"	}"+n+
		"	"+n+
		"	// Synchronise before loading the next tile"+n+
		"	barrier(CLK_LOCAL_MEM_FENCE);"+n+
		"}"+n+
		"// Store the final result in xz"+n+
		"xz[globalCol*x + globalRow] = acc;"+n+
	"}";
	
	/** returns Blob size 32L*x*z which is a multiply of [x][y] by [y][z] */
	public static Blob matmulF(Lazycl lz, Blob xy, Blob yz, int x, int y, int z){
		if(x*y*32 != xy.bize() || y*z*32 != yz.bize()) throw new RuntimeException(
			"Sizes for matmulF not match. xy.bize="+xy.bize()+" yz.bize="+yz.bize()+" x="+x+" y="+y+" z="+z);
		if(x%TS != 0 || y%TS != 0 || z%TS != 0) throw new RuntimeException(
			"All matrix dimensions must be multiples of "+TS+" to use this code. x="+x+" y="+y+" z="+z);
		
		//N was renamed to z
		//PointerBuffer localSize = Lwjgl.pointerBufferOf(TS, (useWPT ? TS/WPT : TS)); 
		//PointerBuffer globalSize = Lwjgl.pointerBufferOf(M, (useWPT ? N/WPT : N));
		
		return lz.lazycl(
			"Code", matmulFCode,
			"Bize", 32L*x*z,
			"GlobalSize", new int[]{x, z},
			"LocalSize", new int[]{TS, TS}, //do matmul in float[TS][TS] x float[TS][TS] blocks in each opencl local mem
			"xy", xy,
			"yz", yz,
			"x", x,
			"y", y,
			"z", z
		);
	}
	
	/** FIXME test if I mixed up the order of dimensions or rows vs cols in matrixs in gpu code of matmulF */
	public Matrix mul(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		if(m1.cols != m2.rows) throw new Error("matrix dimension mismatch");
		final Matrix out = new Matrix(lz, m1.rows, m2.cols);
		out.put("w", matmulF(lz, m1.get("w"), m2.get("w"), m1.rows, m1.cols, m2.cols)); //lazy
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run(){
					//TODO rename m2cols etc to x y z like in the matmulF code
					
					//TODO??? optimize by redesigning lazycl to allow multiple outputs instead of just 1,
					//but would then have to return LazyBlob[] cuz each LazyBlob is still 1 bitstring
					//(could be concat of 2 bitstrings, but then you need to know where to split it, and splitting it has a cost)
					
					m1.put("dw", lz.lazycl(
						"Code",
							"opencl1.2:(global float* m1dwOut, const global float* m1dwIn, const global float* m1w,"+n+
							"		const global float* oDw, const global float* m2w, const int m1cols, const int m2cols){"+n+
							"	int i = get_global_id(0);"+n+
							"	int j = get_global_id(1);"+n+
							"	int outcol = m2cols*i;"+n+
							"	float b = oDw[outcol+j];"+n+
							"	for (int k = 0; k < m1.cols; k++){"+n+
							"		int dwIndex = m1cols*i+k;"+n+
							"		m1dwOut[dwIndex] = m1dwIn[dwIndex] + m2w[(m2cols*k+j] * b;"+n+
							"		//m1dw.putPlus(m1cols*i+k, m2w.f(m2cols*k + j) * b);"+n+
							"	}"+n+
							"}",
						"GlobalSize", new int[]{m1.rows, m2.cols}, //x z
						"m1dwIn", m1.get("dw"),
						"oDw", out.get("dw"),
						"m1w", m1.get("w"),
						"m2w", m2.get("w"),
						"m1cols", m1.cols,
						"m2cols", m2.cols
					));
					m2.put("dw", lz.lazycl(
						"Code",
							"opencl1.2:(global float* m2dwOut, const global float* m2dwIn, const global float* m1w,"+n+
							"		const global float* oDw, const global float* m2w, const int m1cols, const int m2cols){"+n+
							"	int i = get_global_id(0);"+n+
							"	int j = get_global_id(1);"+n+
							"	int outcol = m2cols*i;"+n+
							"	float b = oDw[outcol+j];"+n+
							"	for (int k = 0; k < m1.cols; k++){"+n+
							"		int dwIndex = m2cols*k + j;"+n+
							"		m2dwOut[dwIndex] = m2dwIn[dwIndex] + m1w[m1cols*i+k] * b;"+n+
							"		//m2dw.putPlus(m2cols*k + j, m1w.f(m1cols*i + k) * b);"+n+
							"	}"+n+
							"}",
						"GlobalSize", new int[]{m1.rows, m2.cols}, //x z
						"m2dwIn", m2.get("dw"),
						"oDw", out.get("dw"),
						"m1w", m1.get("w"),
						"m2w", m2.get("w"),
						"m1cols", m1.cols,
						"m2cols", m2.cols
					));
					
					//TODO optimize.
					//m1dw and m2dw dont interfere with eachother, so those 2 things can happen in parallel,
					//which it will automatically after upgrade to not use LazyclPrototype.vm_evalOneTheSlowWayNotInGroups
					//and upgrade to use multiple CLQueue etc.
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix add(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		throw new RuntimeException("TODO");
		/*
		//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Error("matrix dimension mismatch");
		}

		//final Matrix out = new Matrix(lz, lazy, m1.rows, m1.cols);
		final Matrix out = new Matrix(lz, m1.rows, m1.cols);
		
		Blob m1w = m1.get("w");
		Blob m2w = m2.get("w");
		MatrixCache outW = out.cache("w");
		
		for (int i = 0; i < m1.size; i++) {
			outW.put(i, m1w.f(i) + m2w.f(i));
		}
		outW.close();
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable(){
				public void run(){
					Blob outDw = out.get("dw");
					MatrixCache m1dw = m1.cache("dw");
					MatrixCache m2dw = m2.cache("dw");
					for (int i = 0; i < m1.size; i++) {
						float outDwFi = outDw.f(i);
						m1dw.putPlus(i, outDwFi);
						m2dw.putPlus(i, outDwFi);
					}
					MatrixCache.closeAll(m1dw, m2dw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
		*/
	}
	
	/** Example add.rows=200 add.cols=5 rowsOneCol.rows=200 rowsOneCol.cols=1 colMult=5 returns rows=200 cols=5.
	Benrayfields upgrading of recurrentjava to opencl is putting multiple cols as parallelSize
	(unsure if it should be rows or cols yet 2019-5-9, probably cols... UPDATE: 2020-10 whatever the code is now, it works),
	and the bias needs to be added to all parallelIndex vecs, unlike matmul which (it appears) already does.
	Copying and modifying the code from add(...).
	Planning to opencl upgrade after the upgrade to parallelSize and parallelIndex vars.
	<br><br>
	FIXME is this the same as add(Matrix add, Matrix concatVectors colMult of them)? And should it be?
	*/
	public Matrix add_rowsCols_to_rowsColsWithColmult(Matrix add, Matrix rowsOneCol, int colMult){
		Matrix.throwUnless_canBeParamInGraphAgain(add,rowsOneCol);
		throw new RuntimeException("TODO");
		/*
		//boolean lazy = allLazyOrAllNotLazy(add, rowsOneCol);
		if(add.rows != rowsOneCol.rows || add.cols != colMult || rowsOneCol.cols!=1) {
			throw new Error("matrix dimension mismatch or rowsOneCol has more than 1 col");
		}
		//final Matrix out = new Matrix(lz, lazy, add.rows, add.cols);
		final Matrix out = new Matrix(lz, add.rows, add.cols);
		
		MatrixCache outW = out.cache("w");
		Blob addW = add.get("w");
		Blob rowsOneColW = rowsOneCol.get("w");
		
		int offset = 0;
		for(int col=0; col<colMult; col++){
			for (int i = 0; i < rowsOneCol.size; i++){
				outW.put(offset, addW.f(offset) + rowsOneColW.f(i));
				offset++;
			}
		}
		outW.close();
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable(){
				public void run(){
					MatrixCache addDw = add.cache("dw");
					MatrixCache rowsOneColDw = rowsOneCol.cache("dw");
					Blob rowsOneColW = rowsOneCol.get("w");
					Blob outDw = out.get("dw");
					int offset = 0;
					int rowsOneColWSize = rowsOneColW.fsizeIntElseThrow();
					for(int col=0; col<colMult; col++){
						for (int i = 0; i < rowsOneColWSize; i++) {
							
							//m1.dw[i] += out.dw[i];
							addDw.putPlus(offset, outDw.f(offset));
							
							//m2.dw[i] += out.dw[i];
							//FIXME this adjusts the bias colMult times more.
							//Should it be that vs just 1 times as much?
							//If did this with concatVectors of bias, which would it do?
							//Could be a problem if the dws arent changed equally?
							rowsOneColDw.putPlus(i, outDw.f(offset));
							
							offset++;
						}
					}
					MatrixCache.closeAll(addDw, rowsOneColDw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
		*/
	}
	
	public Matrix elmult_rowsCols_to_rowsColsWithColmult(Matrix rowsCols, Matrix rowsOneCol, int colMult){
		Matrix.throwUnless_canBeParamInGraphAgain(rowsCols,rowsOneCol);
		throw new RuntimeException("TODO");
		/*
		//boolean lazy = allLazyOrAllNotLazy(rowsCols, rowsOneCol);
		if (rowsCols.rows != rowsOneCol.rows || rowsCols.cols != colMult || rowsOneCol.cols!=1) {
			throw new Error("matrix dimension mismatch or rowsOneCol has more than 1 col");
		}
		//final Matrix out = new Matrix(lz, lazy, rowsCols.rows, rowsCols.cols);
		final Matrix out = new Matrix(lz, rowsCols.rows, rowsCols.cols);
		
		Blob rowsColsW = rowsCols.get("w");
		Blob rowsOneColW = rowsOneCol.get("w");
		MatrixCache outW = out.cache("w");
		
		int rows = rowsCols.rows;
		int cols = colMult;
		int offset = 0;
		for(int row=0; row<rows; row++){
			for(int col=0; col<rowsCols.cols; col++){
				outW.put(offset, rowsColsW.f(offset) * rowsOneColW.f(row));
				offset++;
			}
		}
		outW.close();
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run(){
					Blob rowsColsW = rowsCols.get("w");
					Blob rowsOneColW = rowsOneCol.get("w");
					Blob outDw = out.get("dw");
					MatrixCache rowsColsDw = rowsCols.cache("dw");
					MatrixCache rowsOneColDw = rowsOneCol.cache("dw");
					int offset = 0;
					for(int row=0; row<rows; row++){
						for(int col=0; col<rowsCols.cols; col++){
							//out.w[offset] = rowsCols.w[offset] * rowsOneCol.w[row];
							rowsColsDw.putPlus(offset, rowsOneColW.f(row) * outDw.f(offset));
							rowsOneColDw.putPlus(row, rowsColsW.f(offset) * outDw.f(offset));
							offset++;
						}
					}
					//for (int i = 0; i < m1.w.length; i++) {
					//	m1.dw[i] += m2.w[i] * out.dw[i];
					//	m2.dw[i] += m1.w[i] * out.dw[i];
					//}
					MatrixCache.closeAll(rowsColsDw, rowsOneColDw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
		*/
	}
	
	public Matrix oneMinus(final Matrix m){
		Matrix.throwUnless_canBeParamInGraphAgain(m);
		throw new RuntimeException("TODO");
		/*
		Matrix ones = Matrix.ones(m.lz, m.rows, m.cols);
		Matrix out = sub(ones, m);
		return out;
		*/
	}
	
	public Matrix sub(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		throw new RuntimeException("TODO");
		/*
		Matrix out = add(m1, neg(m2));
		return out;
		*/
	}
	
	public Matrix smul(final Matrix m, final float s){
		Matrix.throwUnless_canBeParamInGraphAgain(m);
		throw new RuntimeException("TODO");
		/*
		Matrix m2 = Matrix.uniform(lz, m.rows, m.cols, s);
		Matrix out = elmul(m, m2);
		return out;
		*/
	}
		
	public Matrix neg(final Matrix m){
		Matrix.throwUnless_canBeParamInGraphAgain(m);
		throw new RuntimeException("TODO");
		/*
		Matrix negones = Matrix.negones(lz, m.rows, m.cols);
		Matrix out = elmul(negones, m);
		return out;
		*/
	}
	
	public Matrix elmul(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		throw new RuntimeException("TODO");
		/*
		//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Error("matrix dimension mismatch");
		}
		//final Matrix out = new Matrix(lz, lazy, m1.rows, m1.cols);
		final Matrix out = new Matrix(lz, m1.rows, m1.cols);
		
		Blob m1w = m1.get("w");
		Blob m2w = m2.get("w");
		
		MatrixCache outW = out.cache("w");
		for (int i = 0; i < m1.size; i++) {
			outW.put(i, m1w.f(i) * m2w.f(i));
		}
		outW.close();
		
		if (this.applyBackprop) {
			Runnable bp = new Runnable() {
				public void run(){
					Blob outDw = out.get("dw");
					Blob m1w = m1.get("w");
					Blob m2w = m2.get("w");
					MatrixCache outW = out.cache("w");
					MatrixCache m1dw = m1.cache("dw");
					MatrixCache m2dw = m2.cache("dw");
					for (int i = 0; i < m1.size; i++) {
						m1dw.putPlus(i, m2w.f(i) * outDw.f(i));
						m2dw.putPlus(i, m1w.f(i) * outDw.f(i));
					}
					MatrixCache.closeAll(outW, m1dw, m2dw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
		*/
	}
	
	public static Blob elmulF(Lazycl lz, Blob a, Blob b){
		if(a.bize() != b.bize()) throw new RuntimeException("Diff sizes");
		return lz.lazycl(
			"Code",
				"opencl1.2:(global float* out, const global float* a, const global float* b){"+n+
				"	int id = get_global_id(0);"+n+
				"	out[id] = a[id]*b[id];"+n+
				"}",
			"Bize", a.bize(),
			"GlobalSize", new int[]{a.fsizeIntElseThrow()},
			"a", a,
			"b", b
		);
	}
	
	public static Blob elplusF(Lazycl lz, Blob a, Blob b){
		if(a.bize() != b.bize()) throw new RuntimeException("Diff sizes");
		return lz.lazycl(
			"Code",
				"opencl1.2:(global float* out, const global float* a, const global float* b){"+n+
				"	int id = get_global_id(0);"+n+
				"	out[id] = a[id]+b[id];"+n+
				"}",
			"Bize", a.bize(),
			"GlobalSize", new int[]{a.fsizeIntElseThrow()},
			"a", a,
			"b", b
		);
	}
	
	public Matrix[] acyclicFlow(AcyclicFlowF af, Matrix... ins){
		Matrix.throwUnless_canBeParamInGraphAgain(ins);
		throw new Error("TODO");
	}
	
	public void pass(RnnParams params, Consumer<Matrix> outputListener, Consumer<Model> stateResetter,
			Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting){
		throw new RuntimeException("TODO");
		/*
		Trainer.pass(lz, params, outputListener, stateResetter, model, sequences, applyTraining, lossTraining, lossReporting);
		*/
	}
	
	public void updateModelParams(RnnParams p, Model model){
		throw new RuntimeException("TODO");
		/*
		Trainer.updateModelParams(p, model);
		*/
	}

}
