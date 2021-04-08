package immutable.recurrentjava.flop.unary;

import java.io.Serializable;

import immutable.lazycl.spec.Lazycl;
import immutable.util.Blob;
import mutable.recurrentjava.autodiff.LazyclGraph;
import mutable.recurrentjava.matrix.Matrix;

/** benrayfield renamed Nonlinearity to Unaflop (unary floating point op),
and new class Biflop (such as multiply and add) for opencl optimization
redesigning the Graph class to do less and moving some of those ops
into an opencl kernel which will be compiled from these objects.
*/
public interface Unaflop extends Serializable{
	//benrayfield made these public
	public float forward(float x);
	public float deriv(float x);
	
	/** same as many calls of forward, but may be computed in parallel using GPU */
	public Blob forward(Lazycl lz, Blob floats);
	
	/** same as many calls of deriv, but may be computed in parallel using GPU */
	public Blob deriv(Lazycl lz, Blob floats);
	
	/** an optimization of a loop like this: m1dw.putPlus(i, neuron.deriv(m1w.f(i)) * outDw.f(i));
	used in LazyclGraph.nonlin(Unaflop,Matrix),
	IF implemented in subclasses of Unaflop as a single lz.lazycl call,
	instead of the multiple calls (the most general way) in this default implementation.
	TODO optimize implement in subclasses using 1 ndrange kernel (lz.lazycl call) for lower lag.
	*/
	public default Blob x_plus_derivOf_yTimesZ(Lazycl lz, Blob x, Blob y, Blob z){
		Blob mul = LazyclGraph.elmulF(lz, y, z);
		Blob deriv = deriv(lz, mul);
		return LazyclGraph.elplusF(lz, x, deriv);
	}
}
