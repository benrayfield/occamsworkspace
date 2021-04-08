package mutable.recurrentjava.model;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import immutable.lazycl.spec.Lazycl;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.autodiff.Graph;


public class LinearLayer implements Model {
	
	public final Lazycl lz;

	private static final long serialVersionUID = 1L;
	Matrix W;
	//no biases
	
	public LinearLayer(Lazycl lz, int inputDimension, int outputDimension, float initParamsStdDev, Random rng) {
		this.lz = lz;
		W = Matrix.rand(lz, outputDimension, inputDimension, initParamsStdDev, rng);
	}
	
	@Override
	public Matrix forward(Matrix input, Graph g) {
		Matrix out = g.mul(W, input);
		return out;
	}

	@Override
	public void resetState() {

	}

	@Override
	public List<Matrix> getParameters() {
		List<Matrix> result = new ArrayList<>();
		result.add(W);
		return result;
	}
}
