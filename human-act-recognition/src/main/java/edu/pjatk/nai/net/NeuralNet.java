package edu.pjatk.nai.net;

import lombok.Data;
import lombok.val;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


@Data
public class NeuralNet {

    private List<double[][]> layers = new ArrayList<>();

    public void addLayer(double[][] layer) {
        layers.add(layer);
    }

    public void predict(double[] vector, double alpha) {
        double[] out = out(vector, alpha);
        double max = 0;
        double label = 0;
        for (int i = 0; i < out.length; i++) {
            if (out[i] > max) {
                label = i;
                max = out[i];
            }
            System.out.println(String.format("label[%s] with [%s] accuracy", i, out[i]));
        }
        System.out.println(String.format("final prediction [%s]", label));
    }

    public double[] out(double[] vector, double alpha) {
        val outs = outputs(vector, alpha);
        return outs.getRight().get(outs.getRight().size() - 1);
    }

    public Pair<List<double[]>, List<double[]>> outputs(double[] vector, double alpha) {
        List<double[]> inputs = new ArrayList<>();
        inputs.add(vector);
        List<double[]> outs = new ArrayList<>();
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            double[][] layer = layers.get(layerIndex);
            double[] y = new double[layer.length];
            double[] inputVector = inputs.get(inputs.size() - 1);
            for (int neuronIndex = 0; neuronIndex < layer.length; neuronIndex++) {
                double[] neuron = layer[neuronIndex];
                double net = 0;
                for (int weightIndex = 0; weightIndex < neuron.length - 1; weightIndex++) {
                    net += inputVector[weightIndex] * neuron[weightIndex];
                }
                net += neuron[neuron.length - 1]; // incl bias
                y[neuronIndex] = 1. / (1. + Math.pow(Math.E, -alpha * net));
            }
            outs.add(y);
            if (layerIndex != layers.size() - 1) {
                inputs.add(y);
            }
        }
        return Pair.of(inputs, outs);
    }

    public List<double[]> errors(Pair<List<double[]>, List<double[]>> io, double[] desired) {
        List<double[]> inverseErrors = new ArrayList<>();
        double[] lastErrors = new double[layers.get(layers.size() - 1).length];
        for (int nIndex = 0; nIndex < lastErrors.length; nIndex++) {
            double neuronOut = io.getRight().get(io.getRight().size() - 1)[nIndex];
            lastErrors[nIndex] = derivative(neuronOut) * (desired[nIndex] - neuronOut);
        }
        inverseErrors.add(lastErrors);
        if (layers.size() == 1) {
            return inverseErrors;
        }

        for (int i = layers.size() - 2; i >= 0; i--) {
            double[] nextLayerErrors = inverseErrors.get(inverseErrors.size() - 1);
            double[][] nextLayer = layers.get(i+1);
            double[] layerError = new double[layers.get(i).length];
            for (int neuronIndex = 0; neuronIndex < layerError.length; neuronIndex++) {
                double neuronOut = io.getRight().get(i)[neuronIndex];
                double gradient = 0.;
                for (int k = 0; k < nextLayer.length; k++) {
                    gradient += nextLayer[k][neuronIndex] * nextLayerErrors[k];
                }
                layerError[neuronIndex] = derivative(neuronOut) * gradient;
            }
            inverseErrors.add(layerError);
        }
        Collections.reverse(inverseErrors);
        return inverseErrors;
    }

    private double derivative(double out) {
        return out * (1 - out);
    }

    public void update(List<double[]> inputs, List<double[]> errors, double l) {
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            double[][] currentLayer = layers.get(layerIndex);
            double[] input = inputs.get(layerIndex);
            for (int neuronIndex = 0; neuronIndex < currentLayer.length; neuronIndex++) {
                double[] w = currentLayer[neuronIndex];
                for (int weightIndex = 0; weightIndex < w.length -1; weightIndex++) {
                    double wCorrInput = input[weightIndex];
                    double wCorrError = errors.get(layerIndex)[neuronIndex];
                    w[weightIndex] += l * wCorrInput * wCorrError;
                }
                w[w.length - 1] += l * errors.get(layerIndex)[neuronIndex];
            }
        }
    }

    public NeuralNet copy() {
        NeuralNet neuralNet = new NeuralNet();
        for (double[][] layer : layers) {
            double[][] clonedLayer = new double[layer.length][];
            for (int i = 0; i < layer.length; i++) {
                clonedLayer[i] = layer[i].clone();
            }
            neuralNet.addLayer(clonedLayer);
        }
        return neuralNet;
    }
}
