package mutable.recurrentjava.model;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import immutable.lazycl.spec.Lazycl;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.matrix.MatrixCache;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.autodiff.Graph;


/** a constant linearlayer where each output node has weight 1 from only 1 input node,
and the rest of the weights are 0.
*/
public class LastNNodesAreOutput implements Model{

	private static final long serialVersionUID = 1L;
	
	public final int ins, outs;
	
	private Matrix m;
	
	public final Lazycl lz;
	
	public LastNNodesAreOutput(Lazycl lz, int ins, int outs){
		this.lz = lz;
		if(ins < outs) throw new Error("ins < outs");
		this.ins = ins;
		this.outs = outs;
		m = new Matrix(lz, outs, ins);
		
		MatrixCache mW = m.cache("w");
		
		for(int o=0; o<outs; o++){
			int i = ins-outs+o;
			mW.put(ins*o+i, 1f);
		}
		mW.close();
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g){
		//FIXME dont want this to be modified in backprop
		return g.mul(m, input);
	}

	@Override
	public void resetState() {
	}

	@Override
	public List<Matrix> getParameters(){
		return Collections.EMPTY_LIST;
	}
}
