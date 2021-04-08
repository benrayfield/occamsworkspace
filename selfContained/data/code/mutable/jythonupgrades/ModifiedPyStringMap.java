package mutable.jythonupgrades;
import static mutable.util.Lg.*;

import java.util.concurrent.ConcurrentMap;

import org.python.core.PyObject;
import org.python.core.PyStringMap;

public class ModifiedPyStringMap extends PyStringMap{
	
	public PyObject get(PyObject key) {
		PyObject ret = super.get(key);
		lg("pymap get "+key+" -> "+ret);
		return ret;
	}
	
	public PyObject get(PyObject key, PyObject defaultObj){
		PyObject ret = super.get(key,defaultObj);
		lg("pymap get "+key+" "+defaultObj+" -> "+ret);
		return ret;
	}
	
	public ConcurrentMap<Object,PyObject> getMap() {
		ConcurrentMap<Object,PyObject> ret = super.getMap();
		lg("pymap getMap -> "+ret);
		return ret;
	}
	
	

}
