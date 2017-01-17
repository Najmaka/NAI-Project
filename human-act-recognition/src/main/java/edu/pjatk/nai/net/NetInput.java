package edu.pjatk.nai.net;

import lombok.Data;

@Data
public class NetInput {

    /** Attributes of input. */
    private final double[] features;

    /** Desired output vector. */
    private final double[] desired;

    /** Instance label. */
    private final String label;
}
