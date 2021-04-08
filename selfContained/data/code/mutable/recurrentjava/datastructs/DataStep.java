package mutable.recurrentjava.datastructs;
import java.io.Serializable;

import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.matrix.Matrix;


public class DataStep implements Serializable {

	private static final long serialVersionUID = 1L;
	public Matrix input = null;
	public Matrix targetOutput = null;
	public final Lazycl lz;
	
	public DataStep(Lazycl lz){
		this.lz = lz;
	}
	
	public DataStep(Lazycl lz, float[] input, float[] targetOutput) {
		this.lz = lz;
		this.input = new Matrix(lz,input);
		if (targetOutput != null) {
			this.targetOutput = new Matrix(lz,targetOutput);
		}
	}
	
	@Override
	public String toString() {
		
		String result = "";
		//int end = input.buf("w").capacity();
		//FSyMem inputW = input.mem("w");
		Blob inputW = input.get("w");
		int end = inputW.fsizeIntElseThrow();
		for (int i = 0; i < end; i++) {
			result += String.format("%.5f", inputW.f(i)) + "\t";
		}
		result += "\t->\t";
		if (targetOutput != null) {
			//FSyMem targetOutputW = targetOutput.mem("w");
			Blob targetOutputW = targetOutput.get("w");
			end = targetOutputW.fsizeIntElseThrow();
			//for (int i = 0; i < targetOutputW.mem().capacity(); i++) {
			for (int i = 0; i < end; i++) {
				result += String.format("%.5f", targetOutputW.f(i)) + "\t";
			}
		}
		else {
			result += "___\t";
		}
		return result;
	}
}
