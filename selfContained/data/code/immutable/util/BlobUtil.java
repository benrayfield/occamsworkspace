package immutable.util;

public class BlobUtil{
	
	public static void arraycopy(Blob from, int fromIndex, float[] to, int toIndex, int len){
		for(int i=0; i<len; i++) to[toIndex+i] = from.f(fromIndex+i);
	}
	
	public static void arraycopy(Blob from, int fromIndex, double[] to, int toIndex, int len){
		for(int i=0; i<len; i++) to[toIndex+i] = from.d(fromIndex+i);
	}
	
	public static void arraycopy(Blob from, int fromIndex, int[] to, int toIndex, int len){
		for(int i=0; i<len; i++) to[toIndex+i] = from.i(fromIndex+i);
	}
	
	public static void arraycopy(Blob from, int fromIndex, long[] to, int toIndex, int len){
		for(int i=0; i<len; i++) to[toIndex+i] = from.j(fromIndex+i);
	}
	
	public static void arraycopy(Blob from, int fromIndex, byte[] to, int toIndex, int len){
		for(int i=0; i<len; i++) to[toIndex+i] = from.b(fromIndex+i);
	}

}
