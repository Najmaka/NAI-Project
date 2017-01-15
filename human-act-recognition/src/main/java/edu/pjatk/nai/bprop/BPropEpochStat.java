package edu.pjatk.nai.bprop;

import lombok.Data;

@Data
class BPropEpochStat {

    /** Iteration epoch. */
    private final int epoch;

    /** Mean squared error. */
    private final double error;

    /** Overall accuracy over test set. */
    private final double accuracy;

    String csv() {
        return String.format("%s;%s;%s", epoch, error, accuracy);
    }
}
