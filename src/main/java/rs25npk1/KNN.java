package rs25npk1;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.image.FImage;
import org.openimaj.knn.DoubleNearestNeighboursExact;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class KNN implements Classifier {
    private int K, SQUARE_SIZE;
    private String[] classes;
    private DoubleNearestNeighboursExact knn;
    private FeatureExtractor<DoubleFV, FImage> featureExtractor;

    KNN(int k, int square_size) {
        this.K = k;
        this.SQUARE_SIZE = square_size;
    }

    KNN() {
        this(5, 16);
    }

    @Override
    public void train(VFSGroupDataset<FImage> trainingData) {
        // Instance of our feature extractor
        featureExtractor = new TinyImageFeatureExtractor(SQUARE_SIZE);

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
    }

    @Override
    public String classify(FImage f) {
        //TODO make this less awkward
        ArrayList<FImage> r = new ArrayList<>();
        r.add(f);
        return classify(r).get(0);
    }

    private Map<Integer, String> classify(List<FImage> list) {
        int[][] indices = new int[list.size()][K];
        double[][] distances = new double[list.size()][K];
        List<double[]> qus = list.parallelStream().map(i -> featureExtractor.extractFeature(i).values).collect(Collectors.toList());
        knn.searchKNN(qus, K, indices, distances);

        Map<Integer, String> results = new HashMap<>();
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
}
