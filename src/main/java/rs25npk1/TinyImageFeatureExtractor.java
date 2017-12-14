package rs25npk1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

class TinyImageFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
    private int SQUARE_SIZE;

    TinyImageFeatureExtractor(int square_size) {
        this.SQUARE_SIZE = square_size;
    }

    @Override
    public DoubleFV extractFeature(FImage image) {
        // Find the smallest dimension to make a square
        int size = Math.min(image.height, image.width);
        // Extract square from centre of image
        image = image.extractCenter(size, size);
        // Resize
        image.processInplace(new ResizeProcessor(SQUARE_SIZE, SQUARE_SIZE));
        // Return vector from 2D array of pixel values
        DoubleFV feature = new DoubleFV(image.getDoublePixelVector());
        // Zero mean, unit length and return
        //TODO Use DoubleFV.normaliseFV(2) instead of unitLength
        return unitLength(zeroMean(feature));
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

    private DoubleFV unitLength(DoubleFV feature) {
        feature = feature.clone();
        double sum = 0;
        for (double d : feature.values) {
            sum += Math.pow(d, 2);
        }
        sum = Math.sqrt(sum);
        double[] newValues = new double[feature.values.length];
        for (int i = 0; i < feature.values.length; i++) {
            newValues[i] = feature.values[i] / sum;
        }
        return new DoubleFV(newValues);
    }
}