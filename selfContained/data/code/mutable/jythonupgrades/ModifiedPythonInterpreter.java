package mutable.jythonupgrades;
import static mutable.util.Lg.*;

import org.python.core.Py;
import org.python.core.PyObject;
import org.python.core.PyStringMap;
import org.python.util.PythonInterpreter;

/** TODO modify this to, instead of saying "not found" for a function, lookup the listweb json object for
that name and check if it has python code that starts with "def" in it and eval that code and return that function
or call that function or something like that.
*/
public class ModifiedPythonInterpreter extends PythonInterpreter{
	
	public ModifiedPythonInterpreter(){
		super(new ModifiedPyStringMap());
	}
	
	public PyObject eval(PyObject code) {
		PyObject ret = super.eval(code);
		lg("py evalPyobject "+code+" -> "+ret);
		return ret;
	}
	
	public PyObject eval(String s) {
		PyObject ret = super.eval(s);
		lg("py evalString "+s+" -> "+ret);
		return ret;
	}
	
	public PyObject get(String name){
		PyObject ret = super.get(name);
		lg("py getByString "+name+" -> "+ret);
		return ret;
	}
	
	public <T> T get(String name, Class<T> javaclass) {
		T ret = super.get(name, javaclass);
		lg("py getByStringClass "+name+" "+javaclass.getName()+" -> "+ret);
		return ret;
	}

}
