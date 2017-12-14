package rs25npk1;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


class LIN implements OurClassifier {
    LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> localExtractor;
    HardAssigner<double[], double[], IntDoublePair> assigner;
    FeatureExtractor<DoubleFV, FImage> featureExtractor;
    LiblinearAnnotator<FImage, String> ann;
    int patch_size, step_size;

    LIN() {
        this(8, 4);
    }

    LIN(int patch_size, int step_size) {
        this.patch_size = patch_size;
        this.step_size = step_size;
    }

    @Override
    public void train(VFSGroupDataset<FImage> trainingData) {
        localExtractor = new LocalPatchesExtractor(this.patch_size, this.step_size);

        //TODO remove this? splitting should be done before data given to this method
        GroupedRandomSplitter<String, FImage> random = new GroupedRandomSplitter<>(trainingData, 80, 0, 20);

        assigner = makeAssigner(random.getTrainingDataset());

        featureExtractor = new PatchFeatureExtractor(assigner, localExtractor);

        ann = new LiblinearAnnotator<>(featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        System.err.println("TRAINING");
        ann.train(random.getTrainingDataset());
    }

    void run(GroupedDataset<String, ListDataset<FImage>, FImage> testing) {
        //TESTING WITH SMALL TEST SET

        System.err.println("Testing");
        Map<Integer, String> results = new HashMap<>();
        //        for (int i = 0; i < testingData.size(); i++) {
        //            FImage image = testingData.get(i);
        //            String name = testingData.getID(i);

        System.err.println("Classifying images");

        System.out.println(testing.size());
        double correct = 0, total = 0;
        for (String cls : testing.getGroups()) {
            //Loop through each face in the testing set
            for (FImage im : testing.get(cls)) {
                String[] guess = ann.classify(im).getPredictedClasses().toArray(new String[]{});

                StringBuilder guesses = new StringBuilder();
                for (String s : guess) {
                    guesses.append(s);
                    guesses.append(" ");
                }

                if (guesses.toString().equals(cls)) correct++;
                results.put((int) total, guesses.toString());

                total++;
            }
        }

        System.out.println("Correct: " + correct + " | Total: " + total + " | Accuracy: " + (correct / total) * 100);

        System.err.println("Writing");
        try (Writer writer = new FileWriter(new File("run2.txt"))) {
            System.err.println("Classified");
            for (Map.Entry<Integer, String> entry : results.entrySet()) {
                writer.write(String.format("%s %s\n", entry.getKey(), results.get(entry.getKey())));

            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.err.println("Fin!");
    }

    private HardAssigner<double[], double[], IntDoublePair> makeAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset) {
        List<LocalFeature<SpatialLocation, DoubleFV>> patches = new ArrayList<>();

        LocalPatchesExtractor extractor = new LocalPatchesExtractor(8, 4);
        for (FImage i : groupedDataset) {
            patches.addAll(extractor.extractFeature(i));
        }

        List<double[]> keys = new ArrayList<>();
        for (LocalFeature p : patches) {
            keys.add(p.getFeatureVector().asDoubleVector());
        }

        // (try ~500 clusters to start)
        DoubleKMeans dkm = DoubleKMeans.createKDTreeEnsemble(500);

        DoubleCentroidsResult clusters = dkm.cluster(keys.toArray(new double[][]{}));

        return clusters.defaultHardAssigner();
    }

    @Override
    public String classify(FImage f) {
        return null;
    }
}
