package mutable.recurrentjava.autodiff;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import immutable.acyclicflow.AcyclicFlowF;
import immutable.lazycl.spec.Lazycl;
import immutable.rbm.learnloop.OpenclProgs;
import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.rnn.RnnParams;
import immutable.util.Blob;
import immutable.util.MathUtil;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.loss.Loss;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.matrix.MatrixCache;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.trainer.Trainer;
import mutable.util.task.RunnableTask;
import mutable.util.task.Task;

/** a Graph optimized for CPU. A Graph is a mutable builder of numberCrunching ops with backprop (and in some cases also training) built in. */
public strictfp class CpuGraph implements Graph{
	
	public final Lazycl lz;
	
	protected boolean isApplyBackprop;
	public boolean isApplyBackprop(){ return isApplyBackprop; }
	
	/** tasks to do before backprop, if any.
	benrayfield added this to put DependnetOps in,
	the parts that normally happen as soon as a Matrix is created will instead
	be lazyEvaled all at once in opencl, or when opencl is not used then still instant.
	*/
	public List<Task> forwardprop = new ArrayList<>();
	
	//benrayfield made this public for debugging. TODO put DependnetOps in here.
	public List<Task> backpropToDoInReverseOrder = new ArrayList<>();
	
	/** tasks to do for training after forwardprop and backprop, if any */
	public List<Task> trainprop = new ArrayList<>();
	
	public CpuGraph(Lazycl lz){
		this(lz,true);
	}
	
	public CpuGraph(Lazycl lz, boolean applyBackprop){
		this.lz = lz;
		this.isApplyBackprop = applyBackprop;
	}
	
	public void learn(){
		List<Task> tasks = new ArrayList(forwardprop);
		tasks.addAll(MathUtil.reverse(backpropToDoInReverseOrder));
		tasks.addAll(trainprop);
		if(tasks.stream().allMatch(x->(x instanceof RunnableTask))) {
			Task.doTasksInCpu(tasks);
		}else{
			throw new Error("cant Task.doTasksInOpencl(tasks); cuz redesigning to use CpuGraph and OpenclGraph");
		}
		forwardprop.clear();
		backpropToDoInReverseOrder.clear();
		trainprop.clear();
	}
	
	/** (benrayfield made this func) Was it wrong to copy the stepcache?
	I dont see other funcs accessing stepcache places other than in Trainer.
	Maybe thats cuz this op isnt used except inside neural nodes
	and only the Model.getParameters() (such as weights) use stepCache.
	*/
	public Matrix concatVectors(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		if (m1.cols > 1 || m2.cols > 1) {
			throw new Error("Expected column vectors");
		}
		final Matrix out = new Matrix(lz, m1.rows + m2.rows);
		int loc = 0;
		/*boolean doStepCache = m1.hasStepCacheYet() || m2.hasStepCacheYet();
		BiMem<FloatBuffer> outStepCache = out.stepCache;
		BiMem<FloatBuffer> outStepCache = 
		if(m1.hasStepCacheYet() || m2.hasStepCacheYet()){
			out.stepCache();
		}
		BiMem<FloatBuffer> outStepCache = out.stepCache;
		*/
		
		/*
		FSyMem outW = out.mem("w");
		FSyMem outDw = out.mem("dw");
		FSyMem m1w = m1.mem("w");
		FSyMem m1dw = m1.mem("dw");
		FSyMem m2w = m2.mem("w");
		FSyMem m2dw = m2.mem("dw");
		*/
		MatrixCache xoutW = out.cache("w");
		MatrixCache xoutDw = out.cache("dw");
		Blob m1w = m1.get("w");
		Blob m1dw = m1.get("dw");
		Blob m2w = m2.get("w");
		Blob m2dw = m2.get("dw");
		
		for (int i = 0; i < m1.size; i++) {
			xoutW.put(loc, m1w.f(i));
			xoutDw.put(loc, m1dw.f(i));
			//FIXME? out.stepCache[loc] = m1.stepCache[i];
			loc++;
		}
		for (int i = 0; i < m2.size; i++) {
			xoutW.put(loc, m2w.f(i));
			xoutDw.put(loc, m2dw.f(i));
			//FIXME? out.stepCache[loc] = m2.stepCache[i];
			loc++;
		}
		MatrixCache.closeAll(xoutW, xoutDw);
		if(this.isApplyBackprop){
			m1.disable_canBeParamInGraphAgain();
			m2.disable_canBeParamInGraphAgain();
			backpropToDoInReverseOrder.add(new RunnableTask(()->{
				{
					Blob outW = out.get("w");
					Blob outDw = out.get("dw");
					MatrixCache xm1w = m1.cache("w");
					MatrixCache xm1dw = m1.cache("dw");
					MatrixCache xm2w = m2.cache("w");
					MatrixCache xm2dw = m2.cache("dw");
					int locc = 0;
					for (int i = 0; i < m1.size; i++) {
						xm1w.put(i, outW.f(locc));
						xm1dw.put(i, outDw.f(locc));
						//FIXME? m1.stepCache[i] = out.stepCache[loc];
						locc++;
					}
					for (int i = 0; i < m2.size; i++) {
						xm2w.put(i, outW.f(locc));
						xm2dw.put(i, outDw.f(locc));
						//FIXME? m2.stepCache[i] = out.stepCache[loc];
						locc++;
					}
					MatrixCache.closeAll(xm1w, xm1dw, xm2w, xm2dw);
				}
			}));
		}
		return out;
	}
	
	public Matrix nonlin(final Unaflop neuron, final Matrix m){
		Matrix.throwUnless_canBeParamInGraphAgain(m);
		//final Matrix out = new Matrix(lz, m.lazy, m.rows, m.cols);
		final Matrix out = new Matrix(lz, m.rows, m.cols);
		MatrixCache outW = out.cache("w");
		Blob m1w = m.get("w");
		final int n = m.size;
		for (int i = 0; i < n; i++) {
			outW.put(i, neuron.forward(m1w.f(i)));
		}
		MatrixCache.closeAll(outW);
		if (this.isApplyBackprop) {
			Runnable bp = new Runnable() {
				public void run(){
					Blob outDw = out.get("dw");
					Blob m1w = m.get("w");
					MatrixCache m1dw = m.cache("dw");
					for (int i = 0; i < n; i++) {
						m1dw.putPlus(i, neuron.deriv(m1w.f(i)) * outDw.f(i));
					}
					MatrixCache.closeAll(m1dw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	/*public static boolean allLazyOrAllNotLazy(Matrix... m){
		boolean ret = m[0].lazy;
		for(int i=0; i<m.length; i++) if(m[i].lazy != ret) throw new Error("Some lazy and some nonlazy");
		return ret;
	}*/
	
	public Matrix mul(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.cols != m2.rows) {
			throw new Error("matrix dimension mismatch");
		}
		
		final int m1rows = m1.rows;
		final int m1cols = m1.cols;
		final int m2cols = m2.cols;
		//final Matrix out = new Matrix(lz, lazy, m1rows, m2cols);
		final Matrix out = new Matrix(lz, m1rows, m2cols);
		
		Blob m1w = m1.get("w");
		Blob m2w = m2.get("w");
		MatrixCache outW = out.cache("w");
		final int outcols = m2cols;
		for (int i = 0; i < m1rows; i++) {
			int m1col = m1cols*i;
			for (int j = 0; j < m2cols; j++) {
				float dot = 0;
				for (int k = 0; k < m1cols; k++) {
					dot +=  m1w.f(m1col + k) * m2w.f(m2cols*k + j);
				}
				outW.put(outcols*i + j, dot);
			}
		}
		outW.close();
		if (this.isApplyBackprop) {
			Runnable bp = new Runnable() {
				public void run() {
					Blob m1w = m1.get("w");
					Blob m2w = m2.get("w");
					MatrixCache m1dw = m1.cache("dw");
					MatrixCache m2dw = m2.cache("dw");
					Blob outDw = out.get("dw");
					for (int i = 0; i < m1.rows; i++) {
						int outcol = outcols*i;
						for (int j = 0; j < m2.cols; j++) {
							float b = outDw.f(outcol + j);
							for (int k = 0; k < m1.cols; k++) {
								m1dw.putPlus(m1cols*i+k, m2w.f(m2cols*k + j) * b);
								m2dw.putPlus(m2cols*k + j, m1w.f(m1cols*i + k) * b);
							}
						}
					}
					MatrixCache.closeAll(m1dw, m2dw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix add(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Error("matrix dimension mismatch");
		}

		//final Matrix out = new Matrix(lz, lazy, m1.rows, m1.cols);
		final Matrix out = new Matrix(lz, m1.rows, m1.cols);
		
		Blob m1w = m1.get("w");
		Blob m2w = m2.get("w");
		MatrixCache outW = out.cache("w");
		
		for (int i = 0; i < m1.size; i++) {
			outW.put(i, m1w.f(i) + m2w.f(i));
		}
		outW.close();
		
		if (this.isApplyBackprop) {
			Runnable bp = new Runnable(){
				public void run(){
					Blob outDw = out.get("dw");
					MatrixCache m1dw = m1.cache("dw");
					MatrixCache m2dw = m2.cache("dw");
					for (int i = 0; i < m1.size; i++) {
						float outDwFi = outDw.f(i);
						m1dw.putPlus(i, outDwFi);
						m2dw.putPlus(i, outDwFi);
					}
					MatrixCache.closeAll(m1dw, m2dw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	/** Example add.rows=200 add.cols=5 rowsOneCol.rows=200 rowsOneCol.cols=1 colMult=5 returns rows=200 cols=5.
	Benrayfields upgrading of recurrentjava to opencl is putting multiple cols as parallelSize
	(unsure if it should be rows or cols yet 2019-5-9, probably cols... UPDATE: 2020-10 whatever the code is now, it works),
	and the bias needs to be added to all parallelIndex vecs, unlike matmul which (it appears) already does.
	Copying and modifying the code from add(...).
	Planning to opencl upgrade after the upgrade to parallelSize and parallelIndex vars.
	<br><br>
	FIXME is this the same as add(Matrix add, Matrix concatVectors colMult of them)? And should it be?
	*/
	public Matrix add_rowsCols_to_rowsColsWithColmult(Matrix add, Matrix rowsOneCol, int colMult){
		Matrix.throwUnless_canBeParamInGraphAgain(add,rowsOneCol);
		//boolean lazy = allLazyOrAllNotLazy(add, rowsOneCol);
		if(add.rows != rowsOneCol.rows || add.cols != colMult || rowsOneCol.cols!=1) {
			throw new Error("matrix dimension mismatch or rowsOneCol has more than 1 col");
		}
		//final Matrix out = new Matrix(lz, lazy, add.rows, add.cols);
		final Matrix out = new Matrix(lz, add.rows, add.cols);
		
		MatrixCache outW = out.cache("w");
		Blob addW = add.get("w");
		Blob rowsOneColW = rowsOneCol.get("w");
		
		int offset = 0;
		for(int col=0; col<colMult; col++){
			for (int i = 0; i < rowsOneCol.size; i++){
				outW.put(offset, addW.f(offset) + rowsOneColW.f(i));
				offset++;
			}
		}
		outW.close();
		
		if (this.isApplyBackprop) {
			Runnable bp = new Runnable(){
				public void run(){
					MatrixCache addDw = add.cache("dw");
					MatrixCache rowsOneColDw = rowsOneCol.cache("dw");
					Blob rowsOneColW = rowsOneCol.get("w");
					Blob outDw = out.get("dw");
					int offset = 0;
					int rowsOneColWSize = rowsOneColW.fsizeIntElseThrow();
					for(int col=0; col<colMult; col++){
						for (int i = 0; i < rowsOneColWSize; i++) {
							
							//m1.dw[i] += out.dw[i];
							addDw.putPlus(offset, outDw.f(offset));
							
							//m2.dw[i] += out.dw[i];
							//FIXME this adjusts the bias colMult times more.
							//Should it be that vs just 1 times as much?
							//If did this with concatVectors of bias, which would it do?
							//Could be a problem if the dws arent changed equally?
							rowsOneColDw.putPlus(i, outDw.f(offset));
							
							offset++;
						}
					}
					MatrixCache.closeAll(addDw, rowsOneColDw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix elmult_rowsCols_to_rowsColsWithColmult(Matrix rowsCols, Matrix rowsOneCol, int colMult){
		Matrix.throwUnless_canBeParamInGraphAgain(rowsCols,rowsOneCol);
		//boolean lazy = allLazyOrAllNotLazy(rowsCols, rowsOneCol);
		if (rowsCols.rows != rowsOneCol.rows || rowsCols.cols != colMult || rowsOneCol.cols!=1) {
			throw new Error("matrix dimension mismatch or rowsOneCol has more than 1 col");
		}
		//final Matrix out = new Matrix(lz, lazy, rowsCols.rows, rowsCols.cols);
		final Matrix out = new Matrix(lz, rowsCols.rows, rowsCols.cols);
		
		Blob rowsColsW = rowsCols.get("w");
		Blob rowsOneColW = rowsOneCol.get("w");
		MatrixCache outW = out.cache("w");
		
		int rows = rowsCols.rows;
		int cols = colMult;
		int offset = 0;
		for(int row=0; row<rows; row++){
			for(int col=0; col<rowsCols.cols; col++){
				outW.put(offset, rowsColsW.f(offset) * rowsOneColW.f(row));
				offset++;
			}
		}
		outW.close();
		
		if (this.isApplyBackprop) {
			Runnable bp = new Runnable() {
				public void run(){
					Blob rowsColsW = rowsCols.get("w");
					Blob rowsOneColW = rowsOneCol.get("w");
					Blob outDw = out.get("dw");
					MatrixCache rowsColsDw = rowsCols.cache("dw");
					MatrixCache rowsOneColDw = rowsOneCol.cache("dw");
					int offset = 0;
					for(int row=0; row<rows; row++){
						for(int col=0; col<rowsCols.cols; col++){
							//out.w[offset] = rowsCols.w[offset] * rowsOneCol.w[row];
							rowsColsDw.putPlus(offset, rowsOneColW.f(row) * outDw.f(offset));
							rowsOneColDw.putPlus(row, rowsColsW.f(offset) * outDw.f(offset));
							offset++;
						}
					}
					/*for (int i = 0; i < m1.w.length; i++) {
						m1.dw[i] += m2.w[i] * out.dw[i];
						m2.dw[i] += m1.w[i] * out.dw[i];
					}*/
					MatrixCache.closeAll(rowsColsDw, rowsOneColDw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix oneMinus(final Matrix m){
		Matrix.throwUnless_canBeParamInGraphAgain(m);
		Matrix ones = Matrix.ones(m.lz, m.rows, m.cols);
		Matrix out = sub(ones, m);
		return out;
	}
	
	public Matrix sub(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		Matrix out = add(m1, neg(m2));
		return out;
	}
	
	public Matrix smul(final Matrix m, final float s){
		Matrix.throwUnless_canBeParamInGraphAgain(m);
		Matrix m2 = Matrix.uniform(lz, m.rows, m.cols, s);
		Matrix out = elmul(m, m2);
		return out;
	}
		
	public Matrix neg(final Matrix m){
		Matrix negones = Matrix.negones(lz, m.rows, m.cols);
		Matrix out = elmul(negones, m);
		return out;
	}
	
	public Matrix elmul(final Matrix m1, final Matrix m2){
		Matrix.throwUnless_canBeParamInGraphAgain(m1,m2);
		//boolean lazy = allLazyOrAllNotLazy(m1, m2);
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new Error("matrix dimension mismatch");
		}
		//final Matrix out = new Matrix(lz, lazy, m1.rows, m1.cols);
		final Matrix out = new Matrix(lz, m1.rows, m1.cols);
		
		Blob m1w = m1.get("w");
		Blob m2w = m2.get("w");
		
		MatrixCache outW = out.cache("w");
		for (int i = 0; i < m1.size; i++) {
			outW.put(i, m1w.f(i) * m2w.f(i));
		}
		outW.close();
		
		if (this.isApplyBackprop) {
			Runnable bp = new Runnable() {
				public void run(){
					Blob outDw = out.get("dw");
					Blob m1w = m1.get("w");
					Blob m2w = m2.get("w");
					MatrixCache outW = out.cache("w");
					MatrixCache m1dw = m1.cache("dw");
					MatrixCache m2dw = m2.cache("dw");
					for (int i = 0; i < m1.size; i++) {
						m1dw.putPlus(i, m2w.f(i) * outDw.f(i));
						m2dw.putPlus(i, m1w.f(i) * outDw.f(i));
					}
					MatrixCache.closeAll(outW, m1dw, m2dw);
				}
			};
			backpropToDoInReverseOrder.add(new RunnableTask(bp));
		}
		return out;
	}
	
	public Matrix[] acyclicFlow(AcyclicFlowF af, Matrix... ins){
		Matrix.throwUnless_canBeParamInGraphAgain(ins);
		throw new Error("TODO");
	}
	
	public void pass(RnnParams params, Consumer<Matrix> outputListener, Consumer<Model> stateResetter,
			Model model, List<DataSequence> sequences, boolean applyTraining, Loss lossTraining, Loss lossReporting){
		Trainer.pass(lz, params, outputListener, stateResetter, model, sequences, applyTraining, lossTraining, lossReporting);
	}
	
	public void updateModelParams(RnnParams p, Model model){
		Trainer.updateModelParams(p, model);
	}

}
