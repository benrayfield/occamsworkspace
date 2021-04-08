package mutable.recurrentjava.datasets;
import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import immutable.lazycl.spec.Lazycl;
import immutable.recurrentjava.flop.unary.LinearUnit;
import immutable.recurrentjava.flop.unary.Unaflop;
import immutable.util.Blob;
import mutable.dependtask.mem.FSyMem;
import mutable.recurrentjava.autodiff.CpuGraph;
import mutable.recurrentjava.autodiff.Graph;
import mutable.recurrentjava.datastructs.DataSequence;
import mutable.recurrentjava.datastructs.DataSet;
import mutable.recurrentjava.datastructs.DataStep;
import mutable.recurrentjava.util.Util;
import mutable.recurrentjava.loss.LossSoftmax;
import mutable.recurrentjava.matrix.Matrix;
import mutable.recurrentjava.matrix.MatrixCache;
import mutable.recurrentjava.model.Model;


public class TextGeneration extends DataSet{

	public static int reportSequenceLength = 100;
	public static boolean singleWordAutocorrect = false;
	public static boolean reportPerplexity = true;
	private static Map<String, Integer> charToIndex = new HashMap<>();
	private static Map<Integer, String> indexToChar = new HashMap<>();
	private static int dimension;
	private static float[] vecStartEnd;
	private static final int START_END_TOKEN_INDEX = 0;
	private static Set<String> words = new HashSet<>();
	public final Lazycl lz;
	
	public static List<String> generateText(Lazycl lz, Model model, int steps, boolean argmax, float temperature, Random rng) throws Exception{
		List<String> lines = new ArrayList<>();
		//ArMat start = new ArMat(dimension);
		Matrix start = new Matrix(lz, dimension);
		MatrixCache startW = start.cache("w");
		//start.w[START_END_TOKEN_INDEX] = 1f;
		startW.put(START_END_TOKEN_INDEX, 1f);
		model.resetState();
		//Graph g = new Graph(false);
		Graph g = new CpuGraph(lz, false);
		Matrix input = (Matrix)start.clone();
		String line = "";
		for (int s = 0; s < steps; s++) {
			Matrix logprobs = model.forward(input, g);
			Matrix probs = LossSoftmax.getSoftmaxProbs(lz, logprobs, temperature);
			MatrixCache probsW = probs.cache("w");
			
			if (singleWordAutocorrect) {
				Matrix possible = Matrix.ones(lz, dimension, 1);
				Blob possibleW = possible.get("w");
				try {
					possible = singleWordAutocorrect(lz, line);
				}
				catch (Exception e) {
					//TODO: still may be some lingering bugs, so don't constrain by possible if a problem occurs. Fix later..
				}
				float tot = 0;
				//remove impossible transitions
				for (int i = 0; i < probsW.size(); i++) {
					probsW.putMult(i, possibleW.f(i));
					tot += probsW.get(i);
				}
				
				//normalize to sum of 1.0 again
				for (int i = 0; i < probs.size; i++){
					probsW.putDivide(i, tot);
				}
				
				for (int i = 0; i < probs.size; i++) {
					if (probsW.get(i) > 0 && possibleW.f(i) == 0) {
						throw new Exception("Illegal transition");
					}
				}
			}
			
			int indxChosen = -1;
			if (argmax) {
				float high = Float.NEGATIVE_INFINITY;
				for (int i = 0; i < probsW.size(); i++) {
					if (probsW.get(i) > high) {
						high = probsW.get(i);
						indxChosen = i;
					}
				}
			}
			else {
				indxChosen = Util.pickIndexFromRandomVector(probs, rng);
			}
			if (indxChosen == START_END_TOKEN_INDEX) {
				lines.add(line);
				line = "";
				input = (Matrix)start.clone();
				//g = new Graph(false);
				g = new CpuGraph(lz, false);
				model.resetState();
				input = (Matrix)start.clone();
			}
			else {
				String ch = indexToChar.get(indxChosen);
				line += ch;
				MatrixCache inputW = input.cache("w");
				for (int i = 0; i < input.size; i++) {
					inputW.put(i, 0);
				}
				inputW.put(indxChosen, 1f);
				inputW.close();
			}
			probsW.close();
		}
		if (line.equals("") == false) {
			lines.add(line);
		}
		startW.close();
		return lines;
	}
	
	private static Matrix singleWordAutocorrect(Lazycl lz, String sequence) throws Exception {
		
		/*
		 * This restricts the output of the RNN to being composed of words found in the source text.
		 * It makes no attempts to account for probabilities in any way.
		*/
		
		sequence = sequence.replace("\"\n\"", " ");
		if (sequence.equals("") || sequence.endsWith(" ")) { //anything is possible after a space
			return Matrix.ones(lz, dimension, 1);
		}
		String[] parts = sequence.split(" ");
		String lastPartialWord = parts[parts.length-1].trim();
		if (lastPartialWord.equals(" ") || lastPartialWord.contains(" ")) {
			throw new Exception("unexpected");
		}
		List<String> matches = new ArrayList<>();
		for (String word : words) {
			if (word.startsWith(lastPartialWord)) {
				matches.add(word);
			}
		}
		if (matches.size() == 0) {
			throw new Exception("unexpected, no matches for '"+lastPartialWord+"'");
		}
		Matrix result = new Matrix(lz, dimension);
		MatrixCache resultW = result.cache("w");
		boolean hit = false;
		for (String match : matches) {
			if (match.length() < lastPartialWord.length()) {
				throw new Exception("How is match shorter than partial word?");
			}
			if (lastPartialWord.equals(match)) {
				resultW.put(charToIndex.get(" "), 1f);
				resultW.put(START_END_TOKEN_INDEX, 1f);
				continue;
			}
			
			String nextChar = match.charAt(lastPartialWord.length()) + "";
			resultW.put(charToIndex.get(nextChar), 1f);
			hit = true;
		}
		if (hit == false) {
			resultW.put(charToIndex.get(" "), 1f);
			resultW.put(START_END_TOKEN_INDEX, 1f);
		}
		MatrixCache.closeAll(resultW);
		return result;
		
	}
	
	public static String sequenceToSentence(DataSequence sequence) {
		String result = "\"";
		for (int s = 0; s < sequence.steps.size() - 1; s++) {
			DataStep step = sequence.steps.get(s);
			int index = -1;
			Blob stepTargetOutputW = step.targetOutput.get("w");
			int end = stepTargetOutputW.fsizeIntElseThrow();
			for (int i = 0; i < end; i++) {
				if (stepTargetOutputW.f(i) == 1){
					index = i;
					break;
				}
			}
			String ch = indexToChar.get(index);
			result += ch;
		}
		result += "\"\n";
		return result;
	}
	
	public TextGeneration(Lazycl lz, String path) throws Exception {
		this.lz = lz;
		
		System.out.println("Text generation task");
		System.out.println("loading " + path + "...");
		
		File file = new File(path);
		List<String> lines = Files.readAllLines(file.toPath(), Charset.defaultCharset());
		Set<String> chars = new HashSet<>();
		int id = 0;
		
		charToIndex.put("[START/END]", id);
		indexToChar.put(id, "[START/END]");
		id++;
		
		System.out.println("Characters:");
		
		System.out.print("\t");
		
		for (String line : lines) {
			for (int i = 0; i < line.length(); i++) {
				
				String[] parts = line.split(" ");
				for (String part : parts) {
					words.add(part.trim());
				}
				
				String ch = line.charAt(i) + "";
				if (chars.contains(ch) == false) {
					System.out.print(ch);
					chars.add(ch);
					charToIndex.put(ch, id);
					indexToChar.put(id, ch);
					id++;
				}
			}
		}
		
		dimension = chars.size() + 1;
		vecStartEnd = new float[dimension];
		vecStartEnd[START_END_TOKEN_INDEX] = 1f;
		
		List<DataSequence> sequences = new ArrayList<>();
		int size = 0;
		for (String line : lines) {
			List<float[]> vecs = new ArrayList<>();
			vecs.add(vecStartEnd);
			for (int i = 0; i < line.length(); i++) {
				String ch = line.charAt(i) + "";
				int index = charToIndex.get(ch);
				float[] vec = new float[dimension];
				vec[index] = 1f;
				vecs.add(vec);
			}
			vecs.add(vecStartEnd);
			
			DataSequence sequence = new DataSequence();
			for (int i = 0; i < vecs.size() - 1; i++) {
				sequence.steps.add(new DataStep(lz(), vecs.get(i), vecs.get(i+1)));
				size++;
			}
			sequences.add(sequence);
		}
		System.out.println("Total unique chars = " + chars.size());
		System.out.println(size + " steps in training set.");
		
		training = sequences;
		lossTraining = new LossSoftmax(lz);
		lossReporting = new LossSoftmax(lz);
		inputDimension = sequences.get(0).steps.get(0).input.size;
		int loc = 0;
		while (sequences.get(0).steps.get(loc).targetOutput == null) {
			loc++;
		}
		outputDimension = sequences.get(0).steps.get(loc).targetOutput.size;
	}
	
	public Lazycl lz(){
		return lz;
	}

	@Override
	public void DisplayReport(Model model, Random rng) throws Exception {
		System.out.println("========================================");
		System.out.println("REPORT:");
		if (reportPerplexity) {
			System.out.println("\ncalculating perplexity over entire data set...");
			float perplexity = LossSoftmax.calculateMedianPerplexity(lz, model, training);
			System.out.println("\nMedian Perplexity = " + String.format("%.4f", perplexity));
		}
		float[] temperatures = {1f, 0.75f, 0.5f, 0.25f, 0.1f};
		for (float temperature : temperatures) {
			if (TextGeneration.singleWordAutocorrect) {
				System.out.println("\nTemperature "+temperature+" prediction (with single word autocorrect):");
			}
			else {
				System.out.println("\nTemperature "+temperature+" prediction:");
			}
			List<String> guess = TextGeneration.generateText(lz, model, reportSequenceLength, false, temperature, rng);
			for (int i = 0; i < guess.size(); i++) {
				if (i == guess.size()-1) {
					System.out.println("\t\"" + guess.get(i) + "...\"");
				}
				else {
					System.out.println("\t\"" + guess.get(i) + "\"");
				}
				
			}
		}
		if (TextGeneration.singleWordAutocorrect) {
			System.out.println("\nArgmax prediction (with single word autocorrect):");
		}
		else {
			System.out.println("\nArgmax prediction:");
		}
		List<String> guess = TextGeneration.generateText(lz, model, reportSequenceLength, true, 1f, rng);
		for (int i = 0; i < guess.size(); i++) {
			if (i == guess.size()-1) {
				System.out.println("\t\"" + guess.get(i) + "...\"");
			}
			else {
				System.out.println("\t\"" + guess.get(i) + "\"");
			}
			
		}
		System.out.println("========================================");
	}

	@Override
	public Unaflop getModelOutputUnitToUse() {
		return new LinearUnit();
	}
}
