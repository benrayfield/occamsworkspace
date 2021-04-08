package mutable.recurrentjava.matrix;
import java.io.Serializable;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.NavigableMap;
import java.util.NavigableSet;
import java.util.Random;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

import org.lwjgl.BufferUtils;
import org.lwjgl.opencl.CLMem;

import immutable.lazycl.impl.blob.AllFloatConstBlob;
import immutable.lazycl.impl.blob.AllZerosBlob;
import immutable.lazycl.impl.blob.EmptyBlob;
import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;
import mutable.compilers.opencl.lwjgl.LwjglOpenCL;
import mutable.dependtask.DependParam;
import mutable.dependtask.mem.FSyMem;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.common.Rand;
//import mutable.compilers.opencl.connectors.lwjgl.Lwjgl;
import mutable.recurrentjava.RjOptions;

/** Redesigning Matrix to change NavigableMap<String,FSyMem> to NavigableMap<String,immutable.util.Blob>
which is a mutable map of String to immutable Blob of floats. Blob is immutable.
LazyBlob extends Blob to delay calling opencl (many blobs deep) until Blob contents are read.
The purpose of the redesign is to create LazyclGraph in a compatible way with CpuGraph
so both can together be used in GraphTester (or TestGraph or some name like that, TODO)
which will do everything twice in 2 child Graphs and verify they compute the exact same bits,
as a testcase of cpu and gpu computing the exact same bits so I know lazycl works,
and when I know it works I can start using it for LSTM neuralnets and other AI research,
and after that hook lazycl into wikibinator106 as an optimization of its universal function
and port recurrentjava to be made of calls of that universal function.
But first I need to opencl optimize recurrentjava autodiff
which this Matrix class is a core part of.
<br><br>
OLD...
<br><br>
benrayfield changed the float[]s to FMem then generalized to MemInfo
so they could be either FMem (containing FloatBuffer to use in CPU)
or DependParam (containing no memory but to be used with pool of CLMems).
<br><br>
FSyMem (with a DependParam SYmbol) are used instead of some other kind of Mem without a DependParam.
The DependParams are how OpenclUtil.callOpenclDependnet refers to CLMem objects
that a FloatBuffer is not required for but some (inputs and outputs to opencl) have it.
*/
public class Matrix implements Cloneable{
	
	/*TODO redesign Matrix to have a mutable map whose values are immutable Blobs (such as LazyBlob) which are used as float arrays,
	then after that works in CpuGraph, can create the matching LazyclGraph and type of Graph which has multiple Graphs
	that it does the same ops on for testing and verifies they result in the exact same bits (matching cpu and gpu bits).
	
	/*FIXME make Matrix use LazyBlob instead of FSyMem (which contains FloatBuffer) and still use it like LazyBlob.f(int),
	but that would break code that writes to the FSyMem (such as "m1dw.putPlus(m1cols*i+k, m2w.get(m2cols*k + j) * b);"),
	so find a way to make the cpu code and gpu code fit together.
	Need the cpu code at least while testing the gpu code,
	and I'm planning to create a subtype of Graph which contains multiple Graphs and does the same ops on all of them
	and compares the bits of all resulting Matrixs to make sure cpu and gpu compute the same bits,
	and when thats working then use just LazyclGraph.
	*/
	
	//FIXME convert everything back to NavigableMap instead of SortedMap, cuz SortedMap doesnt guarantee a backing SortedSet of keys.

	public final int rows, cols;
	
	/** rows*cols */
	public final int size;
	
	protected boolean canBeParamInGraphAgain = true;
	
	/** becomes false when is 1 of 2 params of Graph.concatVectors(Matrix,Matrix) where Graph.applyBackprop,
	cuz when that backprop happens, this Matrix will have all its Blobs replaced by what concatVectors returned,
	so anything done after concatVectors may or may not,
	depending on mutable.dependtask.DependOp order (which happens automatically in Lazycl),
	do the backprop before or after that copy (between out=concat(Matrix,Matrix) back to out).
	The out and 2 Matrix being concatted should be thought of as 2 backing views of the same Matrix,
	even though they have 2 different locations in memory and the copying between them is delayed.
	Use disableBackprop(); Once false, it never becomes true again.
	*/
	public boolean canBeParamInGraphAgain(){ return true; }
	
	/** see canBeParamInGraphAgain() */
	public void disable_canBeParamInGraphAgain(){
		canBeParamInGraphAgain = false;
	}
	
	/*public void throwUnless_canBeParamInGraphAgain(){
		if(!canBeParamInGraphAgain()) throw new RuntimeException("NOT canBeParamInGraphAgain of this="+this);
	}*/
	
	public static void throwUnless_canBeParamInGraphAgain(Matrix... mx) {
		for(Matrix m : mx) if(!m.canBeParamInGraphAgain()) throw new RuntimeException("NOT canBeParamInGraphAgain of this="+m);
	}
	
	
	//public final boolean lazy;
	
	//public final NavigableMap<String,FSyMem> mems;
	public final NavigableMap<String,Blob> mems;
	
	/** immutable */
	public NavigableSet<String> keys(){
		return Collections.unmodifiableNavigableSet(new TreeSet(mems.keySet()));
	}
	
	/** a Lazycl is stateless other than cache, so you dont have to store this with the Matrix
	and can use any Lazycl when load it back into memory.
	*/
	public final Lazycl lz;
	
	/** Example keys: "w", "dw", "stepCache", but generalizing it to any name for experiments in new neuralnet types *
	public FSyMem mem(String key){
		FSyMem ret = mems.get(key);
		if(ret == null){
			ret = newMem();
			mems.put(key, ret);
		}
		return ret;
	}
	
	/** same key as mem(String key). Creates if not exist. *
	public FloatBuffer buf(String key){
		return mem(key).mem();
	}
	*/
	
	/** symbol, same param as buf(String) and mem(String) *
	public DependParam sy(String key){
		return mem(key).sy;
	}*/
	
	/** unmodifiable but mutable, as mem(String key) can add to this set.
	All params of mem(String key) ever called here, which each created a FSyMem.
	The FSyMem is lazy of creating FloatBuffer as needed but nonlazy of creating DependParam.
	This is useful for naming CLMems andOr FloatBuffers by the same DependParam but not having
	to create duplicate buffers except for inputs and outputs but not most of them which are temp calculations.
	*
	public final SortedSet<String> keys = Collections.unmodifiableSortedSet(mems.navigableKeySet());
	*/
	
	
		
	/** the main data *
	public final FSyMem w;
	
	/** backprop of data *
	public final FSyMem dw;
	
	/** In recurrentjava, only Matrixs that are in a
	mutable.recurrentjava.model.Model.getParameters() use stepCache,
	so this is lazyCreate.
	<br><br>
	Decaying sumOfSquares and L2 norming per weight
	(but in opencl I might use this for a variety of kinds of norming)
	used by mutable.recurrentjava.trainer.Trainer
	in Matrixs returned by
	List<Matrix> mutable.recurrentjava.model.Model.getParameters()
	but is not used in temporary Matrixs like those created in
	mutable.recurrentjava.model.GruLayer.forward(Matrix,Graph).
	*
	public FSyMem stepCache(){
		if(stepCache == null){
			stepCache = newMem();
		}
		return stepCache;
	}
	public boolean hasStepCacheYet(){ return stepCache != null; }
	private FSyMem stepCache;
	*/
	
	/*public FSyMem newMem(){
		return new FSyMem("noComment"+Rand.strongRand.nextLong(), rows*cols);
		/*final int siz = rows*cols;
		return new FSyMem(
			new DependParam(float.class,siz),
			(int size)->BufferUtils.createFloatBuffer(siz)
		);*
		//return lazy ? new DependParam(float.class, rows*cols) : new FMem(rows*cols);
	}*/
	
	public Matrix(Lazycl lz, float[] w){
		this(lz,w.length);
		put("w",lz.wrapb(w)); //wrapb means backing, so caller must not modify it
		//this.mem("w").put(w);
		//FloatBuffer buf = ((SyMem<FloatBuffer>)this.w).mem();
		//this.w.buf.position(0);
		//this.w.buf.put(w);
	}
	
	public Matrix(Lazycl lz, int rows, int cols){
	//public Matrix(boolean lazy, int rows, int cols){
		//this.lazy = lazy;
		this(lz,rows,cols,new TreeMap());
		//w = newMem(); //FIXME use map instead
		//dw = newMem();
	}
	
	public void put(String key, Blob val){
		mems.put(key, val);
	}
	
	/** if !has(key) then returns a Blob of same number of floats as rows*cols=size, but all 0s */
	public Blob get(String key){
		//return mems.getOrDefault(key, EmptyBlob.instance);
		Blob val = mems.get(key);
		if(val == null){
			val = blobOfThisMany0f(size);
			put(key,val);
		}
		return val;
	}
	
	/** these are tiny objects that in abstract math are the given size */
	public static Blob blobOfThisMany0f(int numFloats){
		Blob val = blobOfThisMany0f.get(numFloats);
		if(val == null){
			boolean isTemp = true; //FIXME? isTemp means its not copied from gpu to cpu. Should it be isTemp?
			val = new AllZerosBlob(numFloats*32L, (byte)5, true, isTemp);
			blobOfThisMany0f.put(numFloats, val);
		}
		return val;
	}
	
	/** these are tiny objects that in abstract math are the given size */
	public static Blob blobOfThisMany1f(int numFloats){
		Blob val = blobOfThisMany1f.get(numFloats);
		if(val == null){
			boolean isTemp = true; //FIXME? isTemp means its not copied from gpu to cpu. Should it be isTemp?
			val = new AllFloatConstBlob(numFloats, 1f, isTemp);
			blobOfThisMany1f.put(numFloats, val);
		}
		return val;
	}
	
	/** these are tiny objects that in abstract math are the given size */
	public static Blob blobOfThisManyNeg1f(int numFloats){
		Blob val = blobOfThisMany1f.get(numFloats);
		if(val == null){
			boolean isTemp = true; //FIXME? isTemp means its not copied from gpu to cpu. Should it be isTemp?
			val = new AllFloatConstBlob(numFloats, -1f, isTemp);
			blobOfThisManyNeg1f.put(numFloats, val);
		}
		return val;
	}
	
	private static Map<Integer,Blob> blobOfThisMany0f = new HashMap();
	
	private static Map<Integer,Blob> blobOfThisMany1f = new HashMap();
	
	private static Map<Integer,Blob> blobOfThisManyNeg1f = new HashMap();
	
	public boolean has(String key){
		return mems.containsKey(key);
	}
	
	/** NavigableMap must be mutable */
	public Matrix(Lazycl lz, int rows, int cols, NavigableMap<String,Blob> mems){
	//public Matrix(int rows, int cols, NavigableMap<String,FSyMem> mems){
		this.lz = lz;
		this.rows = rows;
		this.cols = cols;
		this.size = rows*cols;
		this.mems = mems;
	}
	
	/*public Matrix(int rows, int cols, FSyMem w, FSyMem dw, FSyMem stepCache){
		//lazy = w.lazy();
		//if(dw.lazy() != lazy || (stepCache!=null && stepCache.lazy() != lazy))
		//	throw new Error("All must be lazy or all nonlazy");
		this.rows = rows;
		this.cols = cols;
		this.size = rows*cols;
		//this.w = w; //FIXME use map instead
		//this.dw = dw;
		//this.stepCache = stepCache;
	}*/
	
	public Matrix(Lazycl lz, int dim){
		this(lz, dim,1);
	}
	
	public int index(int row, int col){
		return cols*row + col;
	}
	
	public static Matrix uniform(Lazycl lz, int rows, int cols, float s){
		Matrix result = new Matrix(lz, rows, cols);
		//TODO create a Blob type that just stores the 1 float, or 4 bytes or int etc, and repeats that for a given size
		//FSyMem resultW = result.mem("w");
		//for (int i = 0; i < result.size; i++) {
		//	resultW.put(i, s);
		//}
		float[] resultW = new float[rows*cols];
		Arrays.fill(resultW, s);
		result.put("w", lz.wrapb(resultW));
		return result;
	}
	
	public static Matrix ones(Lazycl lz, int rows, int cols) {
		return uniform(lz, rows, cols, 1f);
	}
	
	public static Matrix negones(Lazycl lz, int rows, int cols) {
		return uniform(lz, rows, cols, -1f);
	}
	
	public void normByMaxRadius(float maxRadiusPerRow, float maxRadiusPerCol){
		normByMaxRadius(new MatrixStat(get("w"),rows,cols), maxRadiusPerRow, maxRadiusPerCol);
	}
	
	/** benrayfield is adding funcs to measure and norm, such as by maxradius andOr L1 andOr L2 norm,
	but since theres stepCache (is that a kind of rmsprop?) norming each weight change on bellcurve
	of recent changes to that weight, I'll start with just maxradius since its idempotent of that.
	*/
	public void normByMaxRadius(MatrixStat stat, float maxRadiusPerRow, float maxRadiusPerCol){
		if(maxRadiusPerRow <= 0 || maxRadiusPerCol <= 0) throw new Error("must be positive");
		int offset = 0;
		//FSyMem w = mem("w");
		Blob w = get("w");
		if(w.fsize() > Integer.MAX_VALUE) throw new RuntimeException("Too big: "+w.fsize());
		float[] nextW = new float[(int)w.fsize()];
		for(int c=0; c<cols; c++){
			for(int r=0; r<rows; r++){
				float multCuzOfRow = 1;
				if(maxRadiusPerRow < stat.radiusPerRow[r]){
					multCuzOfRow = maxRadiusPerRow/stat.radiusPerRow[r];
				}
				float multCuzOfCol = 1;
				if(maxRadiusPerCol < stat.radiusPerRow[r]){
					multCuzOfCol = maxRadiusPerCol/stat.radiusPerCol[c];
				}
				//w.putMult(offset, Math.min(multCuzOfRow, multCuzOfCol)); //always multiply by at most 1
				nextW[offset] = w.f(offset)*Math.min(multCuzOfRow,multCuzOfCol); //always multiply by at most 1
				offset++;
			}
		}
		put("w",lz.wrapb(nextW));
	}
	
	public static Matrix rand(Lazycl lz, int rows, int cols, float initParamsStdDev, Random rng) {
		Matrix result = new Matrix(lz, rows, cols);
		//FSyMem resultW = result.mem("w");
		float[] resultW = new float[result.size];
		for (int i = 0; i < result.size; i++) {
			//resultW.put(i, (float)(rng.nextGaussian() * initParamsStdDev));
			resultW[i] = (float)(rng.nextGaussian() * initParamsStdDev);
		}
		result.put("w", lz.wrapb(resultW));
		return result;
	}
	
	public static Matrix ident(Lazycl lz, int dim) {
		Matrix result = new Matrix(lz, dim, dim);
		//FSyMem resultW = result.mem("w");
		float[] resultW = new float[result.size];
		for (int i = 0; i < dim; i++){
			//TODO optimize by creating a func similar to set(int,int,float) but which somehow doesnt call Matrix.mem("w") every time.
			//resultW.put(result.index(i,i), 1f);
			resultW[result.index(i,i)] = 1f;
		}
		result.put("w", lz.wrapb(resultW));
		return result;
	}
	
	/*private void setW(int row, int col, float val) {
		FIXME this could get very inefficient cuz have to get from Map every time
		
		w.put(index(row, col), val);
	}*/
	
	/** Since Blob is immutable, only shallow clone the mutable NavigableMap<String,Blob>.
	<br><br>
	OLD: Clone is deep cuz implemented it that way here, but in java is normally shallow. */
	public Object clone(){
		return new Matrix(lz, rows, cols, new TreeMap(mems));
		/*NavigableMap<String,FSyMem> newMems = new TreeMap();
		for(Map.Entry<String,FSyMem> entry : mems.entrySet()){
			newMems.put(entry.getKey(), (FSyMem)entry.getValue().clone());
		}
		return new Matrix(lz, rows, cols, newMems);
		*/
	}
	
	/** MatrixCache says[
		Since Matrix is a mutable map of String to [immutable Blob of floats],
		and theres lots of existing code which writes floats to Matrix directly
		(before the 2021-3-28 redesign to use Blobs instead of FSyMem/FloatBuffer),
		instead of getting a FSyMem then writing it,
		get one of these and write this,
		and when done writing, close it,
		which does Matrix.put(key, an immutableBlob snapshot of what was mutably written),
		where key is whatever String was used in Matrix.get(String)->Blob.
	]
	TODO keep a map of these until they're closed, instead of a new one each call?
	it would allow multiple callers to write to the same MatrixCache,
	but then would need it to count the number of lockers and not close it until its 0.
	It would help with closing the caches with a single call instead of each individually.
	*/
	public MatrixCache cache(String key){
		return new MatrixCache(this, key);
	}
	
}
