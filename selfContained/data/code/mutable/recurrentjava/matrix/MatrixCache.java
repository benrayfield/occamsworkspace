package mutable.recurrentjava.matrix;

import java.io.Closeable;
import java.io.Flushable;
import java.io.IOException;
import java.nio.FloatBuffer;

import org.lwjgl.BufferUtils;

import immutable.lazycl.impl.LazyclPrototype;
import immutable.lazycl.impl.blob.FloatBufferBlob;
import immutable.util.Blob;
import mutable.compilers.opencl.lwjgl.Lwjgl;
import mutable.compilers.opencl.lwjgl.LwjglOpenCL;
import mutable.dependtask.mem.FSyMem;

/** readable and writable cache of immutable Blob value in a Matrix (which has a map of String to Blob),
and at close() the writes are copied to a Blob and put back in Matrix.
<br><br>
since Matrix is a mutable map of String to [immutable Blob of floats],
and theres lots of existing code which writes floats to Matrix directly
(before the 2021-3-28 redesign to use Blobs instead of FSyMem/FloatBuffer),
instead of getting a FSyMem then writing it,
get one of these and write this,
and when done writing, close it,
which does Matrix.put(key, an immutableBlob snapshot of what was mutably written),
where key is whatever String was used in Matrix.get(String)->Blob.
*/
public class MatrixCache{
	
	public final Matrix matrix;
	
	public final String key;
	
	protected FloatBuffer cache;

	/** content starts as (Blob)matrix.get(key), and is written back there when close. */
	public MatrixCache(Matrix matrix, String key){
		this.matrix = matrix;
		this.key = key;
		this.cache = newMutableFloatbufferCopyOf(matrix.get(key));
	}
	
	/** copy back to Matrix, and this object cant be written again.
	Dont implement Closeable cuz that extends AutoCloseable which may be called automatically
	in a try/catch/finally as it says in AutoCloseable docs but I havent verified as of 2021-3-28.
	*/
	public void close(){
		if(cache != null){
			Blob val = blobSnapshotOf(cache);
			cache = null;
			matrix.put(key, val);
		}
	}
	
	public static FloatBuffer newMutableFloatbufferCopyOf(Blob b){
		//dont use return b.arr(FloatBuffer.class); cuz I'm undecided if that should be mutable or immutable FloatBuffer, backing or nonbacking
		int size = (int)b.fsize();
		if(b.fsize() > Integer.MAX_VALUE) throw new RuntimeException("Too big: "+size+" floats");
		FloatBuffer buf = BufferUtils.createFloatBuffer(size);
		for(int i=0; i<size; i++) buf.put(i, b.f(i));
		return buf;
	}
	
	/** probably is same as Lazycl.wrapb(buf). TODO verify, and should it use that instead? */
	public static Blob blobSnapshotOf(FloatBuffer buf){
		return new FloatBufferBlob(buf);
	}
	
	/** size in floats */
	public int size(){
		return matrix.size;
	}
	
	
	/** FIXME This text was copied from FSyMem:
	read float at int index from the FloatBuffer,
	but this doesnt work if havent queued and executed an opencl action to sync
	from CLMem to FloatBuffer (see OpenclUtil for example,
	but TODO will have a function in Matrix andOr Graph for it.
	*/
	public final float get(int index){
		return cache.get(index);
	}
	
	/** FIXME This text was copied from FSyMem:
	Write float at int index. See comment of get(int).
	*/
	public final void put(int index, float f){
		cache.put(index, f);
	}
	
	/** FIXME This text was copied from FSyMem:
	Write plusEqual f at index. See comment of get(int).
	*/
	public final void putPlus(int index, float addMe){
		cache.put(index, cache.get(index)+addMe);
	}
	
	/** FIXME This text was copied from FSyMem:
	Write multiplyEqual f at index. See comment of get(int).
	*/
	public final void putMult(int index, float multMe){
		cache.put(index, cache.get(index)*multMe);
	}
	
	/** FIXME This text was copied from FSyMem:
	Write divideEqual f at index. See comment of get(int).
	This is probably slightly more accurate than putMult(int, 1/multMe).
	*/
	public final void putDivide(int index, float divideMe){
		cache.put(index, cache.get(index)/divideMe);
	}
	
	public static void closeAll(MatrixCache... caches){
		for(MatrixCache cache : caches) cache.close();
	}
	
	/** copy a range of floats. Like System.arraycopy */
	public static void arraycopy(MatrixCache from, int fromIndex, MatrixCache to, int toIndex, int len){
		FSyMem.arraycopy(from.cache, fromIndex, to.cache, toIndex, len);
	}
	
	/** copy a range of floats. Like System.arraycopy(this,... */
	public static void arraycopy(Blob from, int fromIndex, MatrixCache to, int toIndex, int len){
		FloatBuffer toBuf = to.cache;
		for(int i=0; i<len; i++){
			toBuf.put(toIndex+i, from.f(fromIndex+i));
		}
	}

}
