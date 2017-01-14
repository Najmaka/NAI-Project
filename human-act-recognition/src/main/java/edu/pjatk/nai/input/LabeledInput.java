package edu.pjatk.nai.input;

import lombok.Data;

@Data
public class LabeledInput {
    private final double[] v; // initial input vector
    private final double[] d; // desired output vector
    private final String l; // the label- number
}
