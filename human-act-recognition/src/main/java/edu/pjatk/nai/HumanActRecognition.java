package edu.pjatk.nai;

import edu.pjatk.nai.evaluation.EpochStats;
import edu.pjatk.nai.input.LabeledInput;
import edu.pjatk.nai.input.LabeledInputDataset;
import edu.pjatk.nai.net.LayerFactory;
import edu.pjatk.nai.net.NeuralNet;
import lombok.val;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public class HumanActRecognition {

    public static void main(String[] args) throws Exception {

        Instances instances = loadInstances(new File("datasets/15.arff"));
        List<LabeledInput> inputs = createInputs(instances);
        LabeledInputDataset dataset = new LabeledInputDataset(inputs);

        NeuralNet baseNeuralNet = new NeuralNet();
        baseNeuralNet.addLayer(LayerFactory.create(8, instances.numAttributes()-1));
        baseNeuralNet.addLayer(LayerFactory.create(6, 8));
        baseNeuralNet.addLayer(LayerFactory.create(6, 6));
        baseNeuralNet.addLayer(LayerFactory.create(instances.numClasses(), 6));

        List<EpochStats> stats = new ArrayList<>();

        //todo refactor
        final double ALPHA = 1.;
        final double LEARNING_STEP = 0.01;

        stats.add(stats(ALPHA, baseNeuralNet, dataset.getTesting(), 0));
        for (int i = 1; i <= 1000; i++) {
            List<LabeledInput> trainingLabeledInput = dataset.getTraining();
            Collections.shuffle(trainingLabeledInput);
            for (LabeledInput instance : trainingLabeledInput) {
                val io = baseNeuralNet.outputs(instance.getV(), ALPHA);
                val err = baseNeuralNet.errors(io, instance.getD());
                baseNeuralNet.update(io.getLeft(), err, LEARNING_STEP);
            }
            stats.add(stats(ALPHA, baseNeuralNet, dataset.getTesting(), i));
        }

        System.out.println();
//        AbstractClassifier randomForest = new J48();
//        randomForest.setNumIterations(100);
//        randomForest.buildClassifier(instances);
//        Evaluation evaluation = new Evaluation(instances);
//        evaluation.crossValidateModel(randomForest, instances, 10, new Random(10));
//        System.out.println(evaluation.toSummaryString());
//        System.out.println(evaluation.toMatrixString());
//        System.out.println(evaluation.toClassDetailsString());
    }

    //todo extract to separate class
    private static List<LabeledInput> createInputs(Instances instances) {
        List<LabeledInput> inputs = new ArrayList<>();
        for (Instance instance : instances) {
            double[] vector = new double[instances.numAttributes() - 1]; // omitting decision class
            for (int i = 0; i < vector.length - 1; i++) { // as above
                vector[i] = instance.value(i); // extracting attribute value
            }
            double[] desired = new double[instances.numClasses()];
            desired[(int) instance.classValue()] = 1;
            inputs.add(new LabeledInput(vector, desired, Double.toString(instance.classValue())));
        }
        return inputs;
    }

    //todo extract to separate class
    private static Instances loadInstances(File file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        Instances instances = new Instances(reader);
        instances.setClassIndex(instances.numAttributes() - 1);
        return instances;
    }

    //todo refactor it
    private static EpochStats stats(double alpha, NeuralNet net, List<LabeledInput> li, int epoch) {
        double err = 0.;
        double correct = 0.;
        for (LabeledInput instance : li) {
            double[] out = net.out(instance.getV(), alpha);
            int bestNeuronIndex = 0;
            double bestPrediction = out[0];

            for (int i = 0; i < out.length; i++) {
                if (out[i] > bestPrediction) {
                    bestNeuronIndex = i;
                    bestPrediction = out[i];
                }
            }
            String label = Integer.toString(bestNeuronIndex) + ".0";
            if (Objects.equals(label, instance.getL())) {
                correct++;
            }
            for (int i = 0; i < out.length; i++) {
                err += Math.pow(instance.getD()[i] - out[i], 2);
            }
        }
        EpochStats stats = new EpochStats(epoch, err / li.size(), correct / li.size());
        System.out.println(stats.toString());
        return stats;
    }

}
