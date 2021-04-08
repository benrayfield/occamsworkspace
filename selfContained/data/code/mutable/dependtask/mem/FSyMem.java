package mutable.dependtask.mem;
import java.lang.ref.Reference;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import org.lwjgl.BufferUtils;
import org.lwjgl.PointerBuffer;

import mutable.dependtask.DependParam;
import mutable.dependtask.SyMem;

import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.IntFunction;

public class FSyMem extends SyMem<FloatBuffer> implements Cloneable{
	
	//TODO redesign so the mem is not connected to the DependParam since thats done in OpenclUtil.callOpenclDependnet
	
	public static final IntFunction<PointerBuffer> sizeToPointerBuffer = (int size)->BufferUtils.createPointerBuffer(size);
	
	/** returns FloatBuffer or LongBuffer etc. Cant return PointerBuffer cuz its not a Buffer
	<br><br>
	TODO use SyMem directly instead of its subclass FSyMem, cuz I'm generalizing to more primitive types,
	and I created FSyMem when all the array params I used in opencl were float arrays.
	*
	public static final BiFunction<Class,Integer,Buffer> eltypeAndSizeToBuffer = (Class eltype, Integer size)->{
		if(eltype == float.class) return BufferUtils.createFloatBuffer(size);
		if(eltype == double.class) return BufferUtils.createDoubleBuffer(size);
		if(eltype == int.class) return BufferUtils.createIntBuffer(size);
		if(eltype == long.class) return BufferUtils.createLongBuffer(size);
		if(eltype == byte.class) return BufferUtils.createByteBuffer(size);
		if(eltype == short.class) return BufferUtils.createShortBuffer(size);
		if(eltype == char.class) return BufferUtils.createCharBuffer(size);
		//if(eltype == Reference.class) return BufferUtils.createPointerBuffer(size); //not a Buffer
		throw new RuntimeException("Unknown eltype: "+eltype);
	};*/
	public static final IntFunction<ByteBuffer> sizeInBytesToByteBuffer = (int size)->BufferUtils.createByteBuffer(size);
	
	/** todo use Util.newFloatBuffer(int)? Use sizeInBytesToByteBuffer instead of FloatBuffer, todo. */
	static final IntFunction<FloatBuffer> intToFb = (int siz)->BufferUtils.createFloatBuffer(siz);
	//static final IntFunction<FloatBuffer> intToFb = (int siz)->(FloatBuffer)eltypeAndSizeToBuffer.apply(float.class, siz);
	
	public FSyMem(String comment, int size){
		this(new DependParam(comment, float.class, size));
	}
	
	public FSyMem(DependParam sy){
		this(sy, intToFb);
	}
	
	public FSyMem(DependParam sy, IntFunction<FloatBuffer> memFactory){
		super(sy, memFactory);
		if(sy.elType != float.class) throw new Error("Not a float DependParam");
	}
	
	/** read float at int index from the FloatBuffer,
	but this doesnt work if havent queued and executed an opencl action to sync
	from CLMem to FloatBuffer (see OpenclUtil for example,
	but TODO will have a function in Matrix andOr Graph for it.
	*/
	public final float get(int index){
		//FIXME mem() or mem? mem() might be too slow to call many times
		return mem().get(index);
	}
	
	/** write float at int index. See comment of get(int). */
	public final void put(int index, float f){
		//FIXME mem() or mem? mem() might be too slow to call many times
		mem().put(index,f);
	}
	
	/** write plusEqual f at index. See comment of get(int). */
	public final void putPlus(int index, float addMe){
		//FIXME mem() or mem? mem() might be too slow to call many times
		mem().put(index,mem().get(index)+addMe);
	}
	
	/** write multiplyEqual f at index. See comment of get(int). */
	public final void putMult(int index, float multMe){
		//FIXME mem() or mem? mem() might be too slow to call many times
		mem().put(index,mem().get(index)*multMe);
	}
	
	/** write divideEqual f at index. See comment of get(int).
	This is probably slightly more accurate than putMult(int, 1/multMe).
	*/
	public final void putDivide(int index, float divideMe){
		//FIXME mem() or mem? mem() might be too slow to call many times
		mem().put(index,mem().get(index)/divideMe);
	}
	
	public Object clone(){
		FSyMem m = new FSyMem((DependParam)sy.clone(), memFactory);
		if(mem != null){
			arraycopy(mem, 0, m.mem(), 0, sy.size);
		}
		return m;
	}
	
	public float[] toFloatArray(){
		float[] ret = mem().array().clone();
		if(ret.length != size) throw new Error(
			"FloatBuffer size "+ret.length+" differs from mem() size "+size);
		return ret;
	}
	
	/** Same params as System.arraycopy except for FloatBuffers.
	<br><br>
	TODO optimize by using FloatBuffer.put(FloatBuffer),
	and make sure to put their positions capacities etc
	back the way they were before the copy except the
	range thats been copied.
	*/
	public static void arraycopy(FloatBuffer from, int fromIndex, FloatBuffer to, int toIndex, int len){
		for(int i=0; i<len; i++){
			to.put(toIndex+i, from.get(fromIndex+i));
		}
	}
	
	/** this.writer().accept(Mem) copies that Mem into my Mem */
	public Consumer<Mem> writer(){
		return (Mem m)->{
			if(m instanceof FSyMem){
				copy(((FSyMem)m).mem,mem());
			}else{
				throw new RuntimeException("TODO");
			}
		};
	}
	
	public static void copy(FloatBuffer from, FloatBuffer to){
		from.position(0);
		to.position(0);
		to.put(from);
	}

}
