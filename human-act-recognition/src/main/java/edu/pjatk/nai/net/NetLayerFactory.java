package edu.pjatk.nai.net;

import java.util.concurrent.ThreadLocalRandom;


class NetLayerFactory {

    private static NetLayerFactory instance;

    static NetLayerFactory getInstance() {
        if (instance == null) {
            instance = new NetLayerFactory();
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
