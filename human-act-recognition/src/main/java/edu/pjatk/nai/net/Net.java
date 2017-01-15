package edu.pjatk.nai.net;

import lombok.Data;
import lombok.val;
import org.apache.commons.lang3.tuple.Pair;
import weka.core.Instances;

import java.util.*;


@Data
public class Net {

    /** The alpha parameter used in sigmoid unipolar function. */
    private final double ALPHA;

    /** The list containing all layers. */
    private final List<double[][]> layers = new ArrayList<>();

    /** Maps index of winner neuron from last layer to decision class. */
    private final Map<Integer, String> decisionClasses = new HashMap<>();

    public Net(int[] hiddenLayerSize, Instances instances, double alpha) {
        this.ALPHA = alpha;
        int inputSize = instances.numAttributes() - 1; // minus class attribute
        for (int neuronsNumber : hiddenLayerSize) {
            layers.add(NetLayerFactory.getInstance().create(neuronsNumber, inputSize));
            inputSize = neuronsNumber;
        }
        layers.add(NetLayerFactory.getInstance().create(instances.numClasses(), inputSize));
        for (int i = 0; i < instances.classAttribute().numValues(); i++) {
            decisionClasses.put(i, instances.classAttribute().value(i));
        }
    }

    public String decisionClass(double[] out) {
        int index = 0;
        double best = out[0];
        for (int i = 0; i < out.length; i++) {
            if (out[i] > best) {
                index = i;
                best = out[i];
            }
        }
        return decisionClasses.get(index);
    }

    public double[] out(double[] vector) {
        val outs = outputs(vector);
        return outs.getRight().get(outs.getRight().size() - 1);
    }

    private Pair<List<double[]>, List<double[]>> outputs(double[] vector) {
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
                y[neuronIndex] = 1. / (1. + Math.pow(Math.E, -ALPHA * net));
            }
            outs.add(y);
            if (layerIndex != layers.size() - 1) {
                inputs.add(y);
            }
        }
        return Pair.of(inputs, outs);
    }

    private List<double[]> errors(Pair<List<double[]>, List<double[]>> io, double[] desired) {
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
            double[][] nextLayer = layers.get(i + 1);
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

    private void update(List<double[]> inputs, List<double[]> errors, double l) {
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            double[][] currentLayer = layers.get(layerIndex);
            double[] input = inputs.get(layerIndex);
            for (int neuronIndex = 0; neuronIndex < currentLayer.length; neuronIndex++) {
                double[] w = currentLayer[neuronIndex];
                for (int weightIndex = 0; weightIndex < w.length - 1; weightIndex++) {
                    double wCorrInput = input[weightIndex];
                    double wCorrError = errors.get(layerIndex)[neuronIndex];
                    w[weightIndex] += l * wCorrInput * wCorrError;
                }
                w[w.length - 1] += l * errors.get(layerIndex)[neuronIndex];
            }
        }
    }

    public void backpropagate(NetInput netInput, double learningStep) {
        Pair<List<double[]>, List<double[]>> inputsOutputs = outputs(netInput.getFeatures());
        List<double[]> errors = errors(inputsOutputs, netInput.getDesired());
        update(inputsOutputs.getLeft(), errors, learningStep);
    }
}
