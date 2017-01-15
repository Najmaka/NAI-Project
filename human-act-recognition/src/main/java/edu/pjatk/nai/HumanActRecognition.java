package edu.pjatk.nai;

import edu.pjatk.nai.bprop.BPropRunner;
import edu.pjatk.nai.cli.CliOptions;
import edu.pjatk.nai.cli.CliOptionsParser;
import edu.pjatk.nai.input.InstancesReader;
import edu.pjatk.nai.net.Net;
import org.apache.commons.lang3.tuple.Pair;
import weka.core.Instances;

import java.util.Optional;

public class HumanActRecognition {


    public static void main(String[] args) throws Exception {

        // parsing options given in command line
        CliOptionsParser cliOptionsParser = new CliOptionsParser();
        Optional<CliOptions> optsOptional = cliOptionsParser.parse(args);
        if (!optsOptional.isPresent()) {
            cliOptionsParser.helpAndExit();
        }
        CliOptions opt = optsOptional.get();


        // reading dataset, left (key) are training instances, right (value) are testing ones
        Pair<Instances, Instances> data = new InstancesReader().load(opt.getArff(), opt.getRatio());

        // neural net creation
        Net net = new Net(opt.getLayerSizes(), data.getLeft(), opt.getAlpha());

        // running backpropagation algorithm
        BPropRunner backpropagation = new BPropRunner(opt.getEpochs(), opt.getLearningStep());
        backpropagation.run(net, data.getLeft(), data.getRight());
    }

}
