package edu.pjatk.nai.evaluation;

import lombok.Data;

@Data
public class EpochStats {
    private final int epoch;
    private final double error;
    private final double accuracy;
}
