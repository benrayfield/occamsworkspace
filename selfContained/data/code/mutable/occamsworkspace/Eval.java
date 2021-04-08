package mutable.occamsworkspace;
import static mutable.occamsworkspace.Eval.jython;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import javax.script.Bindings;

import javax.script.ScriptContext;
import javax.script.ScriptEngine;
import javax.script.ScriptException;
import javax.script.SimpleScriptContext;

import org.python.util.PythonInterpreter;

import immutable.util.Text;
import mutable.jythonupgrades.ModifiedPythonInterpreter;
import mutable.util.Files;

//import ch.obermuhlner.scriptengine.jshell.JShellScriptEngine;

public class Eval{
	
	
	
	public static final PythonInterpreter jython;
	static{
		//jython = new PythonInterpreter();
		jython = new ModifiedPythonInterpreter();
		jython.setIn(System.in);
		jython.setErr(System.err);
		jython.setOut(System.out);
	}
	
	
	/*
	public static final ScriptEngine jshell;
	static{
		jshell = new JShellScriptEngine();
		Bindings b = jshell.createBindings();
		//Bindings globalBindings = jshell.createBindings();
		//Bindings engineBindings = jshell.createBindings();
		jshell.setContext(new SimpleScriptContext());
		jshell.setBindings(b, ScriptContext.GLOBAL_SCOPE);
		jshell.setBindings(b, ScriptContext.ENGINE_SCOPE);
		//engine.put("alpha", 2);
		//engine.put("beta", 3);
		//engine.put("gamma", 0);
		//Object result = engine.eval("gamma = alpha + beta");
		//System.out.println("result="+result);
		//System.out.println(engine.eval("Object xyz = 55;"));
		//engine.eval("mutable.occamsworkspace.OWPanel.y = xyz;");
		//System.out.println("y="+y);
		//engine.put("hello", "wORLd");
		//System.out.println(engine.eval("hello+\"zz\""));
	}
	
	/** FIXME 2020-11-25 it can create a class and instantiate it in the same eval call, but after that the class doesnt work. i even saved the class
	in a jshell var of type Class and used reflection on it but it keeps saying "package  does not exist" meaning the default "" package
	since I didnt give the class a specific package. 
	kk=4999999950000000
	duration=0.49199190000000004 loopBodiesPerSec=2.0325537879790297E8
	xy: class REPL.$JShell$3$PairXY
	Exception in thread "main" java.lang.RuntimeException: java.lang.RuntimeException: javax.script.ScriptException: package  does not exist
	Snippet:VariableKey(xy)#2-REPL.$JShell$3.PairXY xy = (REPL.$JShell$3.PairXY) ch.obermuhlner.scriptengine.jshell.VariablesTransfer.getVariableValue("xy");
		at mutable.occamsworkspace.OWPanel.main(OWPanel.java:47)
	Caused by: java.lang.RuntimeException: javax.script.ScriptException: package  does not exist
	Snippet:VariableKey(xy)#2-REPL.$JShell$3.PairXY xy = (REPL.$JShell$3.PairXY) ch.obermuhlner.scriptengine.jshell.VariablesTransfer.getVariableValue("xy");
	*
	public static Object jshell(String code){
		try{
			return jshell.eval(code);
		}catch (ScriptException e){
			throw new RuntimeException(e);
		}
	}
	
	/** set var value *
	public static void jshell(String key, Object val){
		jshell.put(key, val);
	}*/
	
	public static void lg(Object line){
		System.out.println(line.toString());
	}
	
	/** eval python code (or similar to it) in jython
	https://github.com/jython/jython
	*/
	public static Object jy(String code){
		Object ret = null;
		if(!code.startsWith("(lambda") && (code.contains("import") || code.startsWith("def") || code.startsWith("class") || code.contains("="))){
			//fixme could be an = or import in a string literal
			jython.exec(code);
			return null;
		}else{
			return jython.eval(code);
		}
	}
	
	/** eval python code (or similar to it) in jython
	https://github.com/jython/jython
	*
	public static void jython(String code){
		
		//TODO merge jython(String) with jythonEval(String) using the code in OWPanel that checks if it starts with def, class, contains =, etc, to choose which to call.
		
		jython.exec(code);
	}
	
	/** https://github.com/jython/jython *
	public static Object jythonEval(String code){
		return jython.eval(code);
	}*/
	
	/** set var in jython */
	public static void jy(String key, Object val){
		jython.set(key, val);
		
		/*p.exec("print('Hello Python World!'+str(3*4))"); //outputs Hello Python World!12
		//for(int i=0; i<100000; i++){
		//	p.exec("print('Hello Python World!'+str("+i+"*10))");
		//}
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		//String line, prevLine=null;
		String line;
		p.setOut(System.out);
		StringBuilder command = new StringBuilder();
		//p.setIn(System.in);
		while((line = br.readLine()) != null){
			command.append("\r\n").append(line);
			if(line.equals("") && !command.equals("\r\n")){ //push enter twice to eval so can write multiline
				try{
					p.exec(command.toString());
					command.setLength(0);
				}catch(Exception e){
					e.printStackTrace(System.err);
				}
				System.out.println();
			}
		}*/
	}
	
	/** else null */
	public static byte[] getVar(String varName){
		return Files.read(Files.fileOfVar(varName));
	}
	
	/** val==null to delete, else put that val */
	public static void putVar(String varName, byte[] val){
		File f = Files.fileOfVar(varName);
		if(val == null) Files.delete(f);
		else Files.write(f, val);
	}
	
	/** else null. UTF-8 */
	public static String getVarStr(String varName){
		if(varName == null) return null;
		byte[] bytes = getVar(varName);
		if(bytes == null) return null;
		return Text.bytesToStr(bytes);
	}
	
	/** val==null to delete, else put that val. UTF-8 */
	public static void putVarStr(String varName, String val){
		if(varName == null) return;
		byte[] bytesVal = val==null ? null : Text.strToBytes(val);
		putVar(varName, bytesVal);
	}
	
	public static Object jnew(String clazz){
		try{
			return Class.forName(clazz).getConstructor().newInstance();
		}catch(Exception e){
			return new RuntimeException(e);
		}
	}
	
	public static void write(File f, byte[] val){
		f.getParentFile().mkdirs();
		OutputStream out = null;
		try{
			out = new FileOutputStream(f);
			out.write(val);
		}catch(IOException e){
			throw new Error(e);
		}finally{
			if(out != null) try{ out.close(); }catch(IOException e){ throw new Error(e); }
		}
	}

}
