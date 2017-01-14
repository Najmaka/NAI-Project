package edu.pjatk.nai.net;

import java.util.concurrent.ThreadLocalRandom;


public class LayerFactory {

    public static double[][] create(int neuronsNum, int weightsNum) {
        double[][] layer = new double[neuronsNum][];
        for (int i = 0; i < layer.length; i++) {
            layer[i] = new double[weightsNum + 1];
            for (int j = 0; j < weightsNum + 1; j++) {
                layer[i][j] = ThreadLocalRandom.current().nextDouble(2) - 1;
            }
        }
        return layer;
    }
}
