package edu.pjatk.nai.bprop;

import edu.pjatk.nai.net.Net;
import edu.pjatk.nai.net.NetInput;
import edu.pjatk.nai.input.InstancesMapper;
import lombok.Data;
import lombok.val;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Created by vvyk on 15.01.17.
 */
@Data
public class BPropRunner {

    private final double ALPHA;

    private final double LEARNING_STEP;

    public List<BPropEpochStat> run(Net net, Instances train, Instances test) {
        // mapping weka instances to custom neural network input
        List<NetInput> trainInput = new InstancesMapper().toNetInput(train);
        List<NetInput> testInput = new InstancesMapper().toNetInput(test);

        List<BPropEpochStat> epochStats = new ArrayList<>();
        epochStats.add(countStats(ALPHA, net, testInput, 0));
        for (int i = 1; i <= 1000; i++) {
            Collections.shuffle(trainInput);
            for (NetInput netInput : trainInput) {
                val io = net.outputs(netInput.getFeatures(), ALPHA);
                val err = net.errors(io, netInput.getDesired());
                net.update(io.getLeft(), err, LEARNING_STEP);
            }
            epochStats.add(countStats(ALPHA, net, testInput, i));
        }
        return epochStats;
    }

    private BPropEpochStat countStats(double alpha, Net net, List<NetInput> inputs, int epoch) {
        double err = 0.;
        double correct = 0.;
        for (NetInput instance : inputs) {
            double[] out = net.out(instance.getFeatures(), alpha);
            String predictedLabel = net.decisionClass(out);
            if (Objects.equals(predictedLabel, instance.getLabel())) {
                correct++;
            }
            for (int i = 0; i < out.length; i++) {
                err += Math.pow(instance.getDesired()[i] - out[i], 2);
            }
        }
        BPropEpochStat s = new BPropEpochStat(epoch, err / inputs.size(), correct / inputs.size());
        System.out.println(s.toString());
        return s;
    }
}
