package edu.pjatk.nai.input;


import org.apache.commons.lang3.tuple.Pair;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class InstancesReader {

    public Pair<Instances, Instances> load(File file, double ratio) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        Instances dataset = new Instances(reader);
        dataset.setClassIndex(dataset.numAttributes() - 1);
        dataset.randomize(new Random(System.currentTimeMillis()));
        int trainS = (int) Math.round(dataset.numInstances() * ratio);
        int testS = dataset.numInstances() - trainS;
        return Pair.of(new Instances(dataset, 0, trainS), new Instances(dataset, trainS, testS));
    }


}
