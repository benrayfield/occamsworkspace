package mutable.recurrentjava.autodiff;

import immutable.opencl.OpenCL;
import mutable.compilers.opencl.lwjgl.Lwjgl;
import mutable.compilers.opencl.lwjgl.LwjglOpenCL;
import mutable.listweb.todoKeepOnlyWhatUsingIn.humanaicore.wavetree.bit.Bits;

@Deprecated //use LazyclGraph instead
public class TestAutodiffCpuVsGpuMustCalculateSameBits{
	
	/*
	public static void main(String[] args) {
		//TODO CpuGraph vs OpenclGraph (both are Graph). Do pseudorandom Graph ops (with chosen seed so reproducible)
		//and make sure they compute the exact same bits.
		
		//TODO start small, 1 matrix at a time, and work gradually up to gru and lstm neuralnets,
		//all in the same sequence of testcases here.
		
		
		OpenCL cl = LwjglOpenCL.instance();
		boolean backprop = true;
		Graph[] g = {new CpuGraph(backprop), new OpenclGraph(cl,backprop)};
		
		
		
		
		
	}*/
}
