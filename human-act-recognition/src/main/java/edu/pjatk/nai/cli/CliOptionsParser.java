package edu.pjatk.nai.cli;

import org.apache.commons.cli.*;

import java.util.Optional;

/**
 * Created by vvyk on 15.01.17.
 */
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

        //fixme add options parsing

        CliOptions.CliOptionsBuilder cliOptionsBuilder = CliOptions.builder();
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
