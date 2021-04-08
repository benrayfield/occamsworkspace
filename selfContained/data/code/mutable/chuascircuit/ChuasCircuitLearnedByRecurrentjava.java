package mutable.chuascircuit;
import static immutable.util.MathUtil.sigmoid;

import java.util.Random;
import java.util.function.DoubleSupplier;

import immutable.rnn.recurrentjava.MatD;
import immutable.rnn.recurrentjava.RjLearnStep;
//import immutable.rnn.recurrentjava.RjLstm;
import immutable.util.MathUtil;


/** Start with (E) below, the one with the delay,
and display it like B until if its not much accurate
else have it generate its own x y z as in (E) and display that.
It will need a perfect ChuasCircuit over a time range (that delay)
to start but after that will simulate the whole thing in LSTM.
It will need delay added to trainingdata
generated by dataSequence.
OLD[
	There will be 4 ways of scoring the AI, to be solved in this order
	as its approx increasing difficulty but some of the later ones
	Im not sure which are harder. In any case, solving some or all of these
	would give me confidence in the LSTM algorithm so I can get back to mouseai
	after a few harder tests. Any of these should be viewable in ChuasCircuitViewer
	modified to show both correct and observed derivatives...
	(A) start with a sequence it was trained on, and have it literally parrot the outputs back.
	(B) start with a randomChuasCircuitStartState and have the inputs derived by a perfect ChuasCircuit
		and feed those into the AI and have it predict the outputs as compared to
		the outputs of ChuasCircuit. This is invarep (invariant representation)
		since its never been trained on those exact inputs and they chaoticly differ. 
	(C) Start with a randomChuasCircuitStartState and have AI do the outputs and some of
		the next inputs since AI does not see x y and z (sees maybe just x or x and y???)
		so whichever of those it sees will be derived by adding the output mult dt
		since outputs are dxOverDt, dyOverDt, and dzOverDt.
	(D) Same as C except 3 AIs each see 1 (or maybe 2) of the inputs
		so together they cover all the inputs and outputs without the need
		for a perfect ChuasCircuit to fill in any of it.
	(E) Train on all 3 inputs delayed by chaostime (which is different in chuasCircuit
		than for a human chaostime) and outputs are not delayed,
		so a single Lstm.
]
*/
public class ChuasCircuitLearnedByRecurrentjava extends ChuasCircuit{
	
	//See comment of this class for what test to do.
	//Use randomChuasCircuitStartState and dataSequence and delayInputs.
	
	//Use Lstm.java
	
	public final double dtPerAICycle;
	
	protected double dtSync; 
	
	public ChuasCircuitLearnedByRecurrentjava(double dtPerAICycle){
		this.dtPerAICycle = dtPerAICycle;
	}
	
	//Use Lstm.java
	
	public double[] callNeuralnetStatefully(double... ins){
		//TODO throw new Error("TODO");
		return new double[]{.001, .001, .001}; //FIXME
	}
	
	public void nextState(double dt){
		//if(dt != this.dt) throw new Error("Must be same dt as in constructor so neuralnet can learn it");
		
		//FIXME have it learn from just 1 of the 3 inputs,
		//cuz all 3 has no need to remember state in the recurrent (LSTM or GRU) neuralnet.
		
		dtSync += dt;
		while(dtSync >= dtPerAICycle){
			dtSync -= dtPerAICycle;
			double[] ins = new double[]{ z, y, x };
			double[] outs = callNeuralnetStatefully(ins);
			double dzOverDt = outs[0];
			double dyOverDt = outs[1];
			double dxOverDt = outs[2];
			z += dzOverDt*dt;
			y += dyOverDt*dt;
			x += dxOverDt*dt;
		}
	}
	
	/** Returns an array size dataSequence.length-delayCycles.
	where inputs are delayed that much but outputs are not delayed.
	This is for test type (E) in the comment of this class.
	Since LearnStep is in immutable package, shares arrays between them. Dont modify.
	*/
	public RjLearnStep[] delayInputs(RjLearnStep[] dataSequence, int delayCycles){
		//TODO verify delayCycles range
		RjLearnStep[] ret = new RjLearnStep[dataSequence.length-delayCycles];
		for(int i=0; i<ret.length; i++){
			ret[i] = new RjLearnStep(dataSequence[i].ins, dataSequence[i+delayCycles].outs);
		}
		return ret;
	}
	
	/** runs the ChuasCircuit cycles times at dt each. */
	public static RjLearnStep[] dataSequence(ChuasCircuit c, int cycles, float dt){
		RjLearnStep[] ret = new RjLearnStep[cycles];
		int ins = 1, outs = 3;
		double prevX=c.x, prevY=c.y, prevZ=c.z; 
		for(int i=0; i<cycles; i++){
			c.nextState(dt);
			float dxOverDt = (float)(c.x-prevX)/dt;
			float dyOverDt = (float)(c.y-prevY)/dt;
			float dzOverDt = (float)(c.z-prevZ)/dt;
			//TODO sigmoid instead of linear? What is their range?
			ret[i] = new RjLearnStep(
				new MatD(
					ins,
					new float[]{(float)sigmoid(prevX)},
					new float[ins] //stepCache is zeros
				), //FIXME are these backward?
				new MatD(
					outs,
					new float[]{(float)sigmoid(dxOverDt), (float)sigmoid(dyOverDt), (float)sigmoid(dzOverDt)},
					new float[outs] //stepCache is zeros
				) //FIXME are these backward?
			);
		}
		return ret;
	}
	
	/** a state thats likely to be chaotic instead of position get very big or small */
	public static ChuasCircuit randomChuasCircuitStartState(Random r){
		ChuasCircuit c = new ChuasCircuit();
		DoubleSupplier randAdd = ()->(.03*r.nextGaussian());
		DoubleSupplier randMult = ()->(1+randAdd.getAsDouble());
		c.x = .7*randMult.getAsDouble();
		c.y = randAdd.getAsDouble();
		c.z = randAdd.getAsDouble();
		c.c1 = 15.6*randMult.getAsDouble();
		c.c2 = randMult.getAsDouble();
		c.c3 = 28*randMult.getAsDouble();
		c.m0 = -1.143*randMult.getAsDouble();
		c.m1 = -0.714*randMult.getAsDouble();
		return c;
	}

}