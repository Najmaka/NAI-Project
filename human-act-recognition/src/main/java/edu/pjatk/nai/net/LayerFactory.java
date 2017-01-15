package edu.pjatk.nai.net;

import java.util.concurrent.ThreadLocalRandom;


class LayerFactory {

    private static LayerFactory instance;

    static LayerFactory getInstance() {
        if (instance == null) {
            instance = new LayerFactory();
        }
        return instance;
    }

    double[][] create(int layerSize, int inputSize) {
        double[][] layer = new double[layerSize][];
        for (int i = 0; i < layer.length; i++) {
            layer[i] = new double[inputSize + 1]; // including bias
            for (int j = 0; j < inputSize + 1; j++) {
                layer[i][j] = ThreadLocalRandom.current().nextDouble(2) - 1; // weights init
            }
        }
        return layer;
    }
}
