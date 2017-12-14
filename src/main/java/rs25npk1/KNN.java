package rs25npk1;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.util.pair.IntDoublePair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KNN extends OurClassifier {
    private static int K = 5;
    private static int SQUARE_SIZE = 16;
    private static String[] classes;
    private static DoubleNearestNeighboursExact knn;
    private static FeatureExtractor<DoubleFV, FImage> featureExtractor;

    KNN(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {
        super(training, testing);
    }

    void run() {
        // Instance of our feature extractor
        featureExtractor = new TinyImageFeatureExtractor(SQUARE_SIZE);

        // Array of feature vectors
        double[][] features = new double[trainingData.numInstances()][];
        // Array of classes
        classes = new String[trainingData.numInstances()];

        System.out.println("Training");
        AtomicInteger i = new AtomicInteger(0);
        trainingData.forEach((className, imageList) -> {
            imageList.forEach(image -> {
                features[i.get()] = featureExtractor.extractFeature(image).values;
                classes[i.getAndIncrement()] = className;
            });
        });
        trainingData = null;

        knn = new DoubleNearestNeighboursExact(features);
        System.out.println("Testing");
        try (Writer writer = new FileWriter(new File("run1.txt"))) {
            Map<Integer, String> results = classify(testingData);
            System.out.println("Classified");
            IntStream.range(0, testingData.size()).forEach(index -> {
                try {
                    writer.write(String.format("%s %s\n", testingData.getID(index).split("/")[1], results.get(index)));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private Map<Integer, String> classify(List<FImage> list) {
        int[][] indices = new int[list.size()][K];
        double[][] distances = new double[list.size()][K];
        System.out.println("Extracting features");
        List<double[]> qus = list.parallelStream().map(i -> featureExtractor.extractFeature(i).values).collect(Collectors.toList());
        System.out.println("Finding KNN");
        knn.searchKNN(qus, K, indices, distances);

        Map<Integer, String> results = new HashMap<>();
        System.out.println("Finding best class");
        IntStream.range(0, list.size()).forEach(i -> {
            Map<String, Integer> r = new LinkedHashMap<>();
            Arrays.stream(indices[i]).forEach(p -> {
                String c = classes[p];
                int value = r.get(c) == null ? 0 : r.get(c);
                r.put(c, value + 1);
            });
            results.put(i, Collections.max(r.entrySet(), Map.Entry.comparingByValue()).getKey());
        });

        return results;
    }

    @Override
    public ClassificationResult<String> classify(FImage object) {
        return new BasicClassificationResult<String>(2.0);
    }

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
}
