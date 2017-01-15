package edu.pjatk.nai.cli;

import lombok.Builder;
import lombok.Data;

import java.io.File;

/**
 * Created by vvyk on 15.01.17.
 */
@Data
@Builder
public class CliOptions {

    // dataset options
    private final File arff;
    private final double ratio;


    // neural net options
    private final int[] layerSizes;
    private final double alpha;

    // backpropagation options
    private final double learningStep;
    private final int epochs;
}
