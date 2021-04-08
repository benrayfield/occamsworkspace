package mutable.lazycl.ui;

import javax.swing.JPanel;
import javax.swing.JTextArea;

import immutable.lazycl.spec.Lazycl;

/** Type opencl code into a textarea thats a function of (x,y)->colorARGB,
and see it on screen instantly. You can have many dimensions such as (vector[])->colorARGB,
such as vector being 32 floats and 1024x1024 pixels so float[32*1024*1024] input to opencl
and each gpu thread sees 32 of those floats. But I'm going to start with 2 dimensions (x y).
Also, movement and rotation could be done in 32 dimensions using an extra float[32*32] param.
After 2d mandelbrot, try 3d mandelbrot, then maybe more dimensions.
*/
public class FractalPanel extends JPanel{
	
	/*public final Lazycl lz;
	
	public final JTextArea text;
	
	public final JP
	
	public FractalPanel(Lazycl lz, String firstCode){
		this.lz = lz;
		setLayout(new BorderLayout());
	}*/
	
	

}
