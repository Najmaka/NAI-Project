package edu.pjatk.nai.bprop;

import lombok.Data;

@Data
public class BPropEpochStat {
    private final int epoch;
    private final double error;
    private final double accuracy;
}
