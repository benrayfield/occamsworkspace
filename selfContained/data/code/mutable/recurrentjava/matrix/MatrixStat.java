package mutable.recurrentjava.matrix;
//import java.nio.FloatBuffer;
import immutable.util.Blob;

/** stats on a Blob of floats thats normally a Matrix.get("w").
Probably the arrays are only written once, but TODO verify its not some decaying stat.
If its immutable then TODO use a Blob or 2 Blobs instead,
and maybe put those Blobs in the Matrix itself like aMatrix.put("w.stat", newMatrixStat(aMatrix.get("w")))?
But that might keep it longer than would otherwise be garbcoled (garbage collected) so leave it here.
*/
public class MatrixStat{
	
	public final float[] radiusPerRow, radiusPerCol;
	
	//public MatrixStat(float[] w, int rows, int cols){
	//public MatrixStat(FloatBuffer w, int rows, int cols){
	public MatrixStat(Blob w, int rows, int cols){
		//if(rows*cols != w.capacity()) throw new Error("sizes not match");
		if(rows*cols != w.fsize()) throw new Error("sizes not match");
		radiusPerRow = new float[rows];
		radiusPerCol = new float[cols];
		int offset = 0;
		for(int c=0; c<cols; c++){
			for(int r=0; r<rows; r++){
				//float wo = w.get(offset);
				float wo = w.f(offset);
				float sq = wo*wo;
				//float sq = w[offset]*w[offset];
				radiusPerRow[r] += sq;
				radiusPerCol[c] += sq;
				offset++;
			}
		}
		for(int c=0; c<cols; c++){
			radiusPerCol[c] = (float)Math.sqrt(radiusPerCol[c]);
		}
		for(int r=0; r<rows; r++){
			radiusPerRow[r] = (float)Math.sqrt(radiusPerRow[r]);
		}
	}

}
