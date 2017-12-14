package rs25npk1;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;

abstract class OurClassifier {
    static VFSGroupDataset<FImage> trainingData;

    OurClassifier(VFSGroupDataset<FImage> training) {
        trainingData = training;
    }

    abstract void run();

    abstract String classify(FImage f);
}
