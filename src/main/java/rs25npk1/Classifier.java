package rs25npk1;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;

abstract interface Classifier {
    abstract void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingData);
    abstract String classify(FImage f);
}
