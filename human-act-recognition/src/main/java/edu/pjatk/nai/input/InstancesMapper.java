package edu.pjatk.nai.input;

import edu.pjatk.nai.net.NetInput;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.List;

import static java.util.stream.Collectors.toList;

public class InstancesMapper {

    public List<NetInput> toNetInput(Instances instances) {
        return instances.stream()
                        .map(this::toNetInput)
                        .collect(toList());
    }

    private NetInput toNetInput(Instance instance) {
        double[] f = Arrays.copyOfRange(instance.toDoubleArray(), 0, instance.numAttributes() - 1);
        double[] d = new double[instance.classAttribute().numValues()];
        d[(int) instance.classValue()] = 1;
        String l = instance.classAttribute().value((int) instance.classValue());
        return new NetInput(f, d, l);
    }

}
