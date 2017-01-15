package edu.pjatk.nai.bprop;

import com.google.common.collect.Lists;
import edu.pjatk.nai.input.InstancesMapper;
import edu.pjatk.nai.net.Net;
import edu.pjatk.nai.net.NetInput;
import lombok.Data;
import weka.core.Instances;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

@Data
public class BPropRunner {

    private final double LEARNING_STEP;

    public List<BPropEpochStat> run(Net net, Instances train, Instances test) {
        // mapping weka instances to custom neural network input
        List<NetInput> trainInput = new InstancesMapper().toNetInput(train);
        List<NetInput> testInput = new InstancesMapper().toNetInput(test);

        List<BPropEpochStat> stats = Lists.newArrayList(countStats(net, testInput, 0));
        for (int i = 1; i <= 1000; i++) {
            Collections.shuffle(trainInput);
            trainInput.forEach(in -> net.backpropagate(in, LEARNING_STEP));
            stats.add(countStats(net, testInput, i));
        }
        return stats;
    }

    private BPropEpochStat countStats(Net net, List<NetInput> inputs, int epoch) {
        double err = 0.;
        double correct = 0.;
        for (NetInput instance : inputs) {
            double[] out = net.out(instance.getFeatures());
            String predictedLabel = net.decisionClass(out);
            if (Objects.equals(predictedLabel, instance.getLabel())) {
                correct++;
            }
            for (int i = 0; i < out.length; i++) {
                err += Math.pow(instance.getDesired()[i] - out[i], 2);
            }
        }
        final double meanSquaredError = err / inputs.size();
        final double accuracy = correct / inputs.size();
        BPropEpochStat epochStat = new BPropEpochStat(epoch, meanSquaredError, accuracy);
        System.out.println(epochStat.csv());
        return epochStat;
    }
}
