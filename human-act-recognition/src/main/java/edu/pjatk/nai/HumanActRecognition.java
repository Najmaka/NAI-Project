package edu.pjatk.nai;

import edu.pjatk.nai.bprop.BPropRunner;
import edu.pjatk.nai.input.InstancesReader;
import edu.pjatk.nai.net.Net;
import org.apache.commons.lang3.tuple.Pair;
import weka.core.Instances;

import java.io.File;

public class HumanActRecognition {

    public static void main(String[] args) throws Exception {

        // reading dataset, left (key) are training instances, right (value) are testing ones
        File arffFile = new File("datasets/70.arff");
        Pair<Instances, Instances> dataset = new InstancesReader().load(arffFile, 0.9);

        // neural net creation
        Net net = new Net(new int[] {32, 16, 8}, dataset.getLeft(), 1.0);

        // running backpropagation algorithm
        BPropRunner backpropagation = new BPropRunner(0.01);
        backpropagation.run(net, dataset.getLeft(), dataset.getRight());
    }

}
