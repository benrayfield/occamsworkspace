package mutable.occamsworkspace;
import static mutable.occamsworkspace.Eval.*;
/*import java.util.List;
import javax.script.Bindings;
import javax.script.ScriptContext;
import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import javax.script.ScriptException;
import javax.swing.JPanel;
import ch.obermuhlner.scriptengine.jshell.JShellScriptEngine; //https://github.com/eobermuhlner/jshell-scriptengine
*/

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.util.regex.Pattern;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.ScrollPaneConstants;

import immutable.util.Text;

/** todo jython and jshell or something like that */
public class OWPanel extends JPanel{
	
	/*TODO??? use occamsfuncer forest (256 or 512 bit ids including lazyblob) instead of ocget and ocput,
		especially for simple blobs and maps and lists etc, not using its functional ability to its potential yet.
	For example, store in a typeval python code and get it by id and eval it.
	Garbcol all parts of forest not reachable from some set of forest nodes which are marked to keep (including all dependencies).
	The execution of a python etc program would then be (of course not sandboxed in these early experiments)
	a sequence of occamsfuncer objects, and between each of them, some output (also an occamsfuncer object)
	to the user such as display these pixels or sounds or interpretation of game controller movements etc.
	*/
	
	
	
	/*todo see phonedoc for a paragraph about these plans.
	
	todo create key value system similar to Map<String,byte[]> except caches in memory and after a short time saves to harddrive in file of that name
	in dir of same name as the first char (escaped using listweb code), so can put python and java code etc in them and keep going across
	multiple runs of the program.
	Inside that, build 3d mandelbrot (mandelbulb?) using lazycl,
	and after that get back to recurrentjava lazycl optimization but port it to python
	and only use immutable maps, or something like that.
	careful not to try for so much immutable that I end up wanting to finish occamsfuncer before continuing this.
	
	
	todo the key vals go in Files.dataDir/var/xy/xyz or /var/he/hello
	*/
	
	
	
	protected JTextArea textarea;
	
	public OWPanel(){
		/*
		def plus(x,y):
			return x+y
		plus(5,6)
		
		
		jnew('java.util.Random').nextInt()

		> 637142603
		
	
		#tested it with this class from https://vegibit.com/python-class-examples/ and it worked,
		#including x = Vehicle('abc','def','ghi') and x.fuel_level and x.fuel_up() and x.rnd = jnew('java.util.Random') and x.rnd.nextInt()
		class Vehicle:
			def __init__(self, brand, model, type):
				self.brand = brand
				self.model = model
				self.type = type
				self.gas_tank_size = 14
				self.fuel_level = 0 
			def fuel_up(self):
				self.fuel_level = self.gas_tank_size
				print('Gas tank is now full.')
			def drive(self):
				print('The {self.model} is now driving.')


		(lambda x: x*2)(5)
		
		> 10

		*/
		textarea = new JTextArea();
		textarea.setFont(new Font(Font.MONOSPACED, 0, 14));
		textarea.setTabSize(4);
		textarea.setWrapStyleWord(true);
		textarea.setLineWrap(true);
		setLayout(new BorderLayout());
		int h = ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER, v = ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS;
		add(new JScrollPane(textarea,v,h), BorderLayout.CENTER); //TODO use it as jython and javassist code editor
		textarea.addKeyListener(new KeyListener(){
			public void keyTyped(KeyEvent e){}
			public void keyReleased(KeyEvent e){
				String s = textarea.getText();
				s = norm(s);
				if(s.endsWith("\n\n")){
					int lastCommandToRunStartsAt = lastCommandToRunStartsAt(s);
					String command = s.substring(lastCommandToRunStartsAt).trim();
					if(!command.isEmpty()){
						lg("Running command: "+command);
						String response = "";
						try{
							Object ret = jy(command);
							response = ""+ret;
						}catch(Exception ex){
							response = ("ERR: "+ex).trim(); //TODO printStackTrace to string
						}
						textarea.setText(s+"> "+response+"\n");
						textarea.setSelectionStart(textarea.getText().length());
						textarea.setSelectionEnd(textarea.getText().length());
					}
				}
			}
			public void keyPressed(KeyEvent e){}
		});
		textarea.setText("(lambda xy,zz:xy(zz,zz))((lambda a,b:a*b),20)");
		int len = textarea.getText().length();
		textarea.setSelectionStart(0);
		textarea.setSelectionStart(len);
	}
	
	public static String norm(String lines){
		return lines.replace("\r\n", "\n");
	}
	
	/** else throws */
	public static void verifyLines(String lines){
		if(lines.contains("\r")) throw new RuntimeException("Newlines arent normed");
	}
	
	/** returns lines.length() if theres no next command to run aka the last thing is either "> output..."
	or is a command still being typed but hasnt been commanded to run yet by an empty line after it.
	*/
	public static int lastCommandToRunStartsAt(String lines){
		verifyLines(lines); 
		int lastEmptyLine;
		if(lines.endsWith("\n\n") && !lines.trim().contains("\n\n")){ //is first command
			lastEmptyLine = 0;
		}else{
			lastEmptyLine = lines.stripTrailing().lastIndexOf("\n\n");
		}
		int lastOutput = lines.lastIndexOf("\n>");
		int afterLastOutput = lastOutput;
		if(afterLastOutput != -1){
			afterLastOutput = lines.indexOf('\n', lastOutput+"\n>".length());
		}
		if(lastEmptyLine == -1 && lastOutput == -1) return lines.length();
		return Math.max(lastEmptyLine, afterLastOutput);
	}
	
	//public static Object y = null;
	
	public static void main(String[] args){
		
		
		JFrame window = new JFrame("OccamsWorkspace (jython, TODO... javassist, lazycl, occamsfuncer, jsoundcard, occamserver to browser gamepad api, etc)");
		window.setLayout(new BorderLayout());
		OWPanel panel = new OWPanel();
		//panel.setPreferredSize(new Dimension(10000,10000));
		window.add(panel, BorderLayout.CENTER);
		window.setSize(800,700);
		window.setLocation(200,200);
		window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		window.setVisible(true);
		
		try{
			
			lg(jy("2+3"));
			
			jy("hello", "world");
			
			lg(jy("hello+'abc'"));
			
			lg("asdfasdfasdf");
			
			jy(
				"def plus8(x,y):\n"
				+"	return 8+x*y"
				//+"print(plus8(10,10))\n\n"
			);
			
			for(int i=0; i<10; i++) {
				lg(jy("plus8("+i+",100)"));
			}
			
			jy("from java.util import Random");
			jy("r = Random()");
			
			jy("from mutable.occamsworkspace.Eval import jnew");
			jy("g = jnew('java.util.Random')");
			
			lg(jy("r.nextInt()"));
			lg(jy("g.nextInt()"));
			
			jy("from "+Eval.class.getName()+" import putVar as ocput");
			jy("from "+Eval.class.getName()+" import getVar as ocget");
			jy("from "+Eval.class.getName()+" import putVarStr as ocputs");
			jy("from "+Eval.class.getName()+" import getVarStr as ocgets");
			
			
			//lg(jythonEval("java.lang.Random().nextInt()"));
			
			//lg();
			
			/*JavaScriptEngine jav = new JavaScriptEngine();
			String s = "" +
					"public class Script {" +
					"   public String getMessage() {" +
					"	   return \"Hello World\";" +
					"   } " +
					"}";
			//Object result = jav.eval(s);
			//System.out.println("Result: " + result.getClass());
			
			Object x = jav.compile(s);
			System.out.println("x="+x);
			JavaCompiledScript cc = (JavaCompiledScript)x;
			Class cl = cc.getCompiledClass();
			System.out.println("cl: "+cl);
			System.out.println("inst: "+cl.newInstance());
			jav.eval("Object abc = new Script();");
			*/
			
			
			/*
			//https://github.com/eobermuhlner/jshell-scriptengine/issues/5
			
			long start = System.nanoTime();
			jshell("long kk = 0L; for(int i=0; i<100000000; i++) kk += i; System.out.println(\"kk=\"+kk); 20;");
			double duration = (System.nanoTime()-start)*1e-9;
			lg("duration="+duration+" loopBodiesPerSec="+(1e8/duration));
			//Object xy = jshell("public class PairXY{ public final int x, y; public PairXY(int x, int y){ this.x = x; this.y=y; } public Object theT(){ return 9.87.; } }");
			//Object xy = jshell("public class PairXY{ public final int x, y; public PairXY(int x, int y){ this.x = x; this.y=y; } public Object theT(){ return 9.87; } }\n"
			//	+"Object aPair = new PairXY(55,66);");
			//Object xy = jshell("public class PairXY<T>{ public final int x, y; public PairXY(int x, int y){ this.x = x; this.y=y; } public T theT(){ return (T)(Double)9.87; } }\n"
			//	+"Object aPair = new PairXY<Double>(55,66);");
			//Object xy = jshell("class PairXY<T>{ public final int x, y; public PairXY(int x, int y){ this.x = x; this.y=y; } public T theT(){ return (T)(Double)9.87; } } Class PairXY = PairXY.class;");
			Object xy = jshell("class PairXY<T>{ public final int x, y; public PairXY(int x, int y){ this.x = x; this.y=y; } public T theT(){ return (T)(Double)9.87; } } Class PairXY = PairXY.class;");
			lg("xy: "+xy);
			//lg("reflectCallTheT: "+xy.getClass().getMethod("theT").invoke(xy));
			//jshell("Object xy = new PairXY<Double>(100,200);");
			//OWPanel.class.getConstructor().new
			lg("PairXY: "+jshell("PairXY"));
			jshell("Object mm = PairXY.getConstructor(int.class,int.class).newInstance(55,66);");
			lg("mm"+jshell("mm.hashCode()"));
			//Object xy = jshell("class PairXY<T>{ public final int x, y; public PairXY(int x, int y){ this.x = x; this.y=y; } public T theT(){ return 9.87.; } } Object atat = new PairXY(55,66).theT();");
			//new PairXY<Double>(55,66);
			//jshell("Object xy = new PairXY(55,66);");
			//lg("xy="+jshell("xy"));
			//lg(jshell("xy.theT()"));
			*/
			
			
		}catch(Exception e){
			throw new RuntimeException(e);
		}
		
		//jython("long kk = 0L; for(int i=0; i<100000000; i++) kk += i; System.out.println(\"kk=\"+kk); 20;");
		
		/*
		JShell jshell = JShell.create();
		jshell.
		//JShell jshell = JShell.builder().out(System.out).err(System.err).in(System.in).build();
		List<SnippetEvent> list = jshell.eval("int x = 7+3*4;");
		System.out.println("Size of list: " + list.size());
		System.out.println("Value of the expression is : " + list.get(0).value());
		//for(int i=0; i<1000; i++){
		//	Object z = jshell.eval("x").get(0).value();
		//	if(i%100==0) System.out.println(i+": x = "+z);
		//}
		//jshell.eval("mutable.occamsworkspace.OWPanel.y = x;");
		//System.out.println("loop... "+jshell.eval("long kk = 0; for(int i=0; i<1000000; i++) kk += i; System.out.println(\"kk=\"+kk);").get(0).value());
		//jshell.eval("long kk = 0; for(int i=0; i<1000000; i++) kk += i; java.lang.System.out.println(\"kk=\"+kk);");
		jshell.drop(new Snippet())
		jshell.eval("print(\"abc\");");
		jshell.eval("long kk = 0; for(int i=0; i<1000000; i++) kk += i; print\"kk=\"+kk);");
		*/
		
		
		/*try{
			//ScriptEngineManager manager = new ScriptEngineManager();
			//ScriptEngine engine = manager.getEngineByName("jshell");
			ScriptEngine engine = new JShellScriptEngine();
			Bindings b = engine.createBindings();
			engine.setBindings(b, ScriptContext.GLOBAL_SCOPE);
			engine.put("alpha", 2);
			engine.put("beta", 3);
			engine.put("gamma", 0);
			Object result = engine.eval("gamma = alpha + beta");
			System.out.println("result="+result);
			System.out.println(engine.eval("Object xyz = 55;"));
			
			
			engine.eval("mutable.occamsworkspace.OWPanel.y = xyz;");
			System.out.println("y="+y);
			engine.put("hello", "wORLd");
			System.out.println(engine.eval("hello+\"zz\""));
			
			
			////engine.eval("long kk = 0L; for(int i=0; i<1000000; i++){ kk += i; }");
			//engine.eval("long kk = 0; for(int i=0; i<1000000; i++) kk += i; System.out.println(\"kk=\"+kk); 20;");
			//engine.eval("kk = 0L; for(int i=0; i<10000000; i++) kk += i; System.out.println(\"kk=\"+kk); 20;");
			//long start = System.nanoTime();
			//engine.eval("kk = 0L; for(int i=0; i<100000000; i++) kk += i; System.out.println(\"kk=\"+kk); 20;");
			//double duration = (System.nanoTime()-start)*1e-9;
			//System.out.println("duration="+duration+" loopBodiesPerSec="+(1e8/duration));
			////engine.eval("kk = 0L; for(int i=0; i<1000000000; i++) kk += i; System.out.println(\"kk=\"+kk); 20;");
			////engine.eval("long kk = 0; for(int i=0; i<1000000; i++) kk += i;");
			////engine.eval("long kk = 0; for(int i=0; i<1000000; i++){ kk += i; } return 456;");
			////engine.eval("int kk = 0;", b);
			
		}catch(ScriptException e){ throw new RuntimeException(e); }
		*/
	}

}
