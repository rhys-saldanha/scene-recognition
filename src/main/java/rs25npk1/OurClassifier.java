package rs25npk1;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;

abstract interface OurClassifier {
    abstract void train(VFSGroupDataset<FImage> trainingData);
    abstract String classify(FImage f);
}
