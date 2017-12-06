package rs25npk1;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;

import java.io.*;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Run1 {

    private static int K = 15;
    private static int SQUARE_SIZE = 16;
    private static String[] classes;
    private static DoubleNearestNeighboursExact knn;
    private static FeatureExtractor<DoubleFV, FImage> featureExtractor;
    private static VFSGroupDataset<FImage> trainingData;
    private static VFSListDataset<FImage> testingData;

    public static void main(String[] args) throws FileSystemException, URISyntaxException {
        initialise_data();

        // Instance of our feature extractor
        featureExtractor = new TinyImageFeatureExtractor();

        // Array of feature vectors
        double[][] features = new double[trainingData.numInstances()][];
        // Array of classes
        classes = new String[trainingData.numInstances()];

        AtomicInteger i = new AtomicInteger(0);
        trainingData.forEach((className, imageList) -> {
            imageList.forEach(image -> {
                features[i.get()] = featureExtractor.extractFeature(image).values;
                classes[i.getAndIncrement()] = className;
            });
        });

        knn = new DoubleNearestNeighboursExact(features);

        try (PrintWriter writer = new PrintWriter("run1.txt", "UTF-8")) {
            testingData.parallelStream().forEach(image -> {
                int index = testingData.indexOf(image);
                String result = String.format("%s %s", testingData.getID(index).split("/")[1], classify(image));
//                System.out.println(result);
                writer.println(result);
            });
        } catch (FileNotFoundException | UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

    private static void initialise_data() {
        URL training = ClassLoader.getSystemResource("training");
        URL testing = ClassLoader.getSystemResource("testing");

        String trainingURI = null;
        if (training == null) {
            System.out.println("Using training URL");
            trainingURI = "zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip";
        } else {
            try {
                trainingURI = training.toURI().getPath();
            } catch (URISyntaxException e) {
                e.printStackTrace();
            }
        }

        String testingURI = null;
        if (testing == null) {
            System.out.println("Using testing URL");
            testingURI = "zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip";
        } else {
            try {
                testingURI = testing.toURI().getPath();
            } catch (URISyntaxException e) {
                e.printStackTrace();
            }
        }

        try {
            trainingData = new VFSGroupDataset<FImage>(trainingURI, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
        // Remove training folder
        trainingData.remove("training");

        try {
            testingData = new VFSListDataset<>(testingURI, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
    }

    private static String classify(FImage randomImage) {
        List<IntDoublePair> nn = knn.searchKNN(featureExtractor.extractFeature(randomImage).values, K);
        Map<String, Integer> results = new HashMap<>();

//        nn.forEach(p -> System.out.println(p.toString()));
        nn.forEach(p -> {
            String c = classes[p.getFirst()];
            int value = results.get(c) == null ? 0 : results.get(c);
            results.put(c, value + 1);
        });
//        results.forEach((k, v) -> System.out.println(String.format("%s, %s", k, v)));
        return Collections.max(results.entrySet(), Map.Entry.comparingByValue()).getKey();
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
}
