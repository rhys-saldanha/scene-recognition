package rs25npk1;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.util.array.ArrayUtils;

import java.net.URISyntaxException;
import java.net.URL;

public class Run1 {

    private static final int K = 15;
    private static final int SQUARE_SIZE = 16;

    public static void main(String[] args) throws FileSystemException, URISyntaxException {
        //TODO use URLs, fucking Nick
        URL testing = ClassLoader.getSystemResource("testing.zip");
        URL training = ClassLoader.getSystemResource("training.zip");

        GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingData = new VFSGroupDataset<>("zip:" + training.toURI().getPath(), ImageUtilities.FIMAGE_READER);
        VFSListDataset<FImage> testingData = new VFSListDataset<>("zip:" + testing.toURI().getPath(), ImageUtilities.FIMAGE_READER);

        DisplayUtilities.display(trainingData.getRandomInstance(), "training data random image");
        DisplayUtilities.display(testingData.getRandomInstance(), "testing data random image");
    }

    static class TinyImageFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {

        @Override
        public DoubleFV extractFeature(FImage image) {
            // Find the smallest dimension to make a square
            int size = Math.min(image.height, image.width);
            // Extract square from centre of image
            image = image.extractCenter(size, size);
            // Resize
            image.processInplace(new ResizeProcessor(SQUARE_SIZE, SQUARE_SIZE));
            // Return vector from 2D array of pixel values
            DoubleFV feature = new DoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(image.pixels)));
            // Zero mean, unit length and return
            return (zeroMean(feature)).normaliseFV();
        }

        private DoubleFV zeroMean(DoubleFV feature) {
            feature = feature.clone();
            double mean = 0;
            for (double d : feature.values) {
                mean += d;
            }
            mean /= feature.values.length;
            double[] newValues = new double[feature.values.length];
            for (int i = 0; i < feature.values.length; i++) {
                newValues[i] = feature.values[i] - mean;
            }
            return new DoubleFV(newValues);
        }
    }
}
