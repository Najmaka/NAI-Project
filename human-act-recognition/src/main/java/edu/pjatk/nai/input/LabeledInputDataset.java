package edu.pjatk.nai.input;

import lombok.Getter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by vvyk on 14.01.17.
 */
@Getter
public class LabeledInputDataset {
    private List<LabeledInput> training = new ArrayList<>();
    private List<LabeledInput> testing = new ArrayList<>();

    public LabeledInputDataset(List<LabeledInput> inputs) {
        List<LabeledInput> shuffledInputs = new ArrayList<>(inputs);
        Collections.shuffle(shuffledInputs);
        int bound = (int) (shuffledInputs.size() * 0.8);
        for (int i = 0; i < shuffledInputs.size(); i++) {
            if (i < bound) {
                training.add(shuffledInputs.get(i));
            } else {
                testing.add(shuffledInputs.get(i));
            }
        }
    }
}
