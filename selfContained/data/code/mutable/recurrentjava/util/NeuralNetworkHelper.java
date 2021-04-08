package mutable.recurrentjava.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import immutable.lazycl.spec.Lazycl;
import immutable.recurrentjava.flop.unary.Unaflop;
import mutable.recurrentjava.model.FeedForwardLayer;
import mutable.recurrentjava.model.GruLayer;
import mutable.recurrentjava.model.LastNNodesAreOutput;
import mutable.recurrentjava.model.LinearLayer;
import mutable.recurrentjava.model.LstmLayer;
import mutable.recurrentjava.model.Model;
import mutable.recurrentjava.model.NeuralNetwork;
import mutable.recurrentjava.model.RnnLayer;

public class NeuralNetworkHelper {
	
	public static NeuralNetwork makeLstm(Lazycl lz, int parallelSize, int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Unaflop decoderUnit, float initParamsStdDev, Random rng) {
		List<Model> layers = new ArrayList<>();
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new LstmLayer(lz, parallelSize, inputDimension, hiddenDimension, initParamsStdDev, rng));
			}
			else {
				layers.add(new LstmLayer(lz, parallelSize, hiddenDimension, hiddenDimension, initParamsStdDev, rng));
			}
		}
		layers.add(new FeedForwardLayer(lz, hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
		return new NeuralNetwork(layers);
	}
	
	public static NeuralNetwork makeLstmWithInputBottleneck(Lazycl lz, int parallelSize, int inputDimension, int bottleneckDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Unaflop decoderUnit, float initParamsStdDev, Random rng) {
		List<Model> layers = new ArrayList<>();
		layers.add(new LinearLayer(lz, inputDimension, bottleneckDimension, initParamsStdDev, rng));
		for (int h = 0; h < hiddenLayers; h++){
			if (h == 0){
				layers.add(new LstmLayer(lz, parallelSize, bottleneckDimension, hiddenDimension, initParamsStdDev, rng));
			}
			else {
				layers.add(new LstmLayer(lz, parallelSize, hiddenDimension, hiddenDimension, initParamsStdDev, rng));
			}
		}
		layers.add(new FeedForwardLayer(lz, hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
		return new NeuralNetwork(layers);
	}
	
	public static NeuralNetwork makeFeedForward(Lazycl lz, int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Unaflop hiddenUnit, Unaflop decoderUnit, float initParamsStdDev, Random rng) {
		List<Model> layers = new ArrayList<>();
		if (hiddenLayers == 0) {
			layers.add(new FeedForwardLayer(lz, inputDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
			return new NeuralNetwork(layers);
		}
		else {
			for (int h = 0; h < hiddenLayers; h++) {
				if (h == 0) {
					layers.add(new FeedForwardLayer(lz, inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
				}
				else {
					layers.add(new FeedForwardLayer(lz, hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
				}
			}
			layers.add(new FeedForwardLayer(lz, hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
			return new NeuralNetwork(layers);
		}
	}
	
	/** benrayfield added this. Same as other makeGru func except can have multiple feedforward layers after the GRU.
	Adding this cuz the outputs learn relative position well but dont seem to learn the average size.
	If feedforwardSizes.length==1 this is the same as the other makeGru func. 
	*/
	public static NeuralNetwork makeGru(Lazycl lz, int parallelSize, int inputDimension, int hiddenDimension, int hiddenLayers, int[] feedforwardSizes, Unaflop decoderUnit, float initParamsStdDev, Random rng) {
		List<Model> layers = new ArrayList<>();
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new GruLayer(lz, parallelSize, inputDimension, hiddenDimension, initParamsStdDev, rng));
			}
			else {
				layers.add(new GruLayer(lz, parallelSize, hiddenDimension, hiddenDimension, initParamsStdDev, rng));
			}
		}
		for(int i=0; i<feedforwardSizes.length; i++){
			int inputSize = i==0 ? hiddenDimension : feedforwardSizes[i-1];
			int outputSize = feedforwardSizes[i];
			layers.add(new FeedForwardLayer(lz, inputSize, outputSize, decoderUnit, initParamsStdDev, rng));
		}
		return new NeuralNetwork(layers);
	}
	
	/** benrayfield added this */
	public static NeuralNetwork makeGruWithLinearOutLayer(
			Lazycl lz, int parallelSize, int ins, int hiddens, int outs, float initGruStdDev, float initLinearStdDev, Random rng){
		return new NeuralNetwork(Arrays.<Model>asList(
			new GruLayer(lz, parallelSize, ins, hiddens, initGruStdDev, rng),
			new LinearLayer(lz, hiddens, outs, initLinearStdDev, rng)
		));
	}
	
	/** benrayfield added this */
	public static NeuralNetwork makeGruWithLastNNodesAsOutput(
			Lazycl lz, int parallelSize, int ins, int hiddens, int outs, float initGruStdDev, Random rng){
		return new NeuralNetwork(Arrays.<Model>asList(
			new GruLayer(lz, parallelSize, ins, hiddens, initGruStdDev, rng),
			new LastNNodesAreOutput(lz, hiddens, outs)
		));
	}
	
	public static NeuralNetwork makeGru(Lazycl lz, int parallelSize, int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Unaflop decoderUnit, float initParamsStdDev, Random rng) {
		List<Model> layers = new ArrayList<>();
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new GruLayer(lz, parallelSize, inputDimension, hiddenDimension, initParamsStdDev, rng));
			}
			else {
				layers.add(new GruLayer(lz, parallelSize, hiddenDimension, hiddenDimension, initParamsStdDev, rng));
			}
		}
		layers.add(new FeedForwardLayer(lz, hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
		return new NeuralNetwork(layers);
	}
	
	public static NeuralNetwork makeRnn(Lazycl lz, int inputDimension, int hiddenDimension, int hiddenLayers, int outputDimension, Unaflop hiddenUnit, Unaflop decoderUnit, float initParamsStdDev, Random rng) {
		List<Model> layers = new ArrayList<>();
		for (int h = 0; h < hiddenLayers; h++) {
			if (h == 0) {
				layers.add(new RnnLayer(lz, inputDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
			}
			else {
				layers.add(new RnnLayer(lz, hiddenDimension, hiddenDimension, hiddenUnit, initParamsStdDev, rng));
			}
		}
		layers.add(new FeedForwardLayer(lz, hiddenDimension, outputDimension, decoderUnit, initParamsStdDev, rng));
		return new NeuralNetwork(layers);
	}
}
