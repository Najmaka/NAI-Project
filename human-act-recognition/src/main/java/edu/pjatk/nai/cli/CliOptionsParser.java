package edu.pjatk.nai.cli;

import org.apache.commons.cli.*;

import java.io.File;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toList;


public class CliOptionsParser {

    private final Options options;

    public CliOptionsParser() {
        options = new Options();
        options.addOption("f", true, "dataset arff file path (string)");
        options.addOption("r", true, "dataset instances ratio (double)");
        options.addOption("l", true, "hidden layers size, eg. \"16,16,8\" (string)");
        options.addOption("a", true, "alpha parameter value (double)");
        options.addOption("n", true, "learning step value (double)");
        options.addOption("e", true, "number of epochs (double)");
    }


    public Optional<CliOptions> parse(String[] args) throws ParseException {

        // in case of empty arguments
        if (args.length == 0) {
            helpAndExit();
        }

        // parsing the options
        CommandLineParser parser = new DefaultParser();
        CommandLine commandLine = parser.parse(options, args);

        // validating required options (all) presence
        if (!validateRequiredOptions(commandLine)) {
            return Optional.empty();
        }

        CliOptions.CliOptionsBuilder cliOptionsBuilder = CliOptions.builder();
        cliOptionsBuilder.alpha(Double.parseDouble(commandLine.getOptionValue("a")));
        cliOptionsBuilder.arff(new File(commandLine.getOptionValue("f")));
        cliOptionsBuilder.epochs(Integer.parseInt(commandLine.getOptionValue("e")));
        cliOptionsBuilder.learningStep(Double.parseDouble(commandLine.getOptionValue("n")));
        cliOptionsBuilder.ratio(Double.parseDouble(commandLine.getOptionValue("r")));
        String[] strIntegers = commandLine.getOptionValue("l").split(",");
        List<Integer> layersSize = Stream.of(strIntegers).map(Integer::parseInt).collect(toList());
        int[] sizes = new int[layersSize.size()];
        for (int i = 0; i < layersSize.size(); i++) {
            sizes[i] = layersSize.get(i);
        }
        cliOptionsBuilder.layerSizes(sizes);

        return Optional.of(cliOptionsBuilder.build());
    }

    private boolean validateRequiredOptions(CommandLine commandLine) {
        for (Option option : options.getOptions()) {
            if (!commandLine.hasOption(option.getOpt())) {
                return false;
            }
        }
        return true;
    }

    public void helpAndExit() {
        new HelpFormatter().printHelp("All below options must be provided", options);
        System.exit(1);
    }
}
