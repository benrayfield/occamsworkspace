package mutable.util;

import immutable.lazycl.impl.LazyclPrototype;
import immutable.lazycl.spec.Lazycl;

public class Options{
	private Options(){}
	
	public static Lazycl defaultLazycl(){
		Lazycl lz = LazyclPrototype.instance();
		System.out.println("defaultLazycl returning "+lz);
		return lz;
	}

}
