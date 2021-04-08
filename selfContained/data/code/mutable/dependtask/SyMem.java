package mutable.dependtask;

import java.nio.FloatBuffer;
import java.util.function.IntFunction;

import mutable.dependtask.mem.FSyMem;
import mutable.dependtask.mem.Mem;

/** symbol (DependParam) and memory.
The symbol is used in OpenclUtil.callOpenclDependnet
to refer to memory that is used by potentially multiple DependOps
in the same dependnet of DependOps before returning from GPU to CPU.
Doing multiple opencl kernel calls before returning to CPU extremely reduces lag.
It reads andOr writes a pooled CLMem multiple times
when that symbol is used in multiple DependOp,
and a single DependOp may read andOr write that.
For example, 50 times per second, do 30 opencl ndrange kernels in 1 callOpenclDependnet.
*/
public class SyMem<T> extends Mem{
	
	//TODO redesign so the mem is not connected to the DependParam since thats done in OpenclUtil.callOpenclDependnet

	public final DependParam sy;
	
	/** such as FloatBuffer or float[] or CLMem but CLMem normally doesnt go here cuz
	is encapsulated inside the OpenCL.java interface which takes Mem in params and includes them in return.
	TODO also DoubleBuffer, IntBuffer, LongBuffer, etc.
	*/
	protected T mem;
	
	/** Param is size in units of elType such as floats.
	At most once, creates memory such as FloatBuffer.
	For many Mems, only the DependParam is ever used such as to
	abstractly define but not get a copy of internal opencl calculations.
	*/
	protected IntFunction<T> memFactory;
	
	/** lazy allocates FloatBuffer, LongBuffer, etc depending on DependParam.elType and DependParam.size */
	public SyMem(DependParam sy){
		//Example: T is FloatBuffer.class or LongBuffer.class.
		//this(sy, (IntFunction<T>)((int size)->(T)FSyMem.eltypeAndSizeToBuffer.apply(sy.elType,sy.size)));
		this(
			sy,
			//FIXME might be expecting DoubleBuffer but get ByteBuffer of 8 times the size but in units of bytes instead of doubles,
			//which is what I want it to return but the FIXME is to not expect Buffer types other than ByteBuffer,
			//so SyMem<ByteBuffer> is ok and SyMem<double[]> is ok but SyMem<DoubleBuffer> is not ok,
			//cuz ByteBufferBlob can efficiently read byte short char int float long and double (and never writes cuz is immutable).
			(IntFunction)((int size)->FSyMem.sizeInBytesToByteBuffer.apply(sy.byteSize()))
		);
	}
	
	/** Example "T buf": a FloatBuffer or IntBuffer */
	public SyMem(DependParam sy, T buf){
		super(sy.elType, sy.size);
		this.sy = sy;
		this.mem = buf;
		this.memFactory = null;
	}
	
	public SyMem(DependParam sy, IntFunction<T> memFactory){
		super(sy.elType, sy.size);
		this.sy = sy;
		this.memFactory = memFactory;
	}
	
	/** after calling this once, can use this.mem */
	public T mem(){
		if(mem == null) mem = memFactory.apply(sy.size);
		return mem;
	}
	
	public void put(float[] a){
		if(mem() instanceof FloatBuffer){
			((FloatBuffer)mem).position(0);
			((FloatBuffer)mem).put(a);
		}else{
			throw new UnsupportedOperationException();
		}
	}

}
