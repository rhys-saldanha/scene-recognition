package rs25npk1;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
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
import java.util.*;


public class LIN extends OurClassifier {
    LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> localExtractor;
    HardAssigner<double[], double[], IntDoublePair> assigner;
    FeatureExtractor<DoubleFV, FImage> featureExtractor;
    LiblinearAnnotator<FImage, String> ann;
    int patch_size, step_size;

    public static void main(String[] args) {
        (new LIN(8, 4)).run();
    }

    LIN(int patch_size, int step_size) {
        this.patch_size = patch_size;
        this.step_size = step_size;
    }

    @Override
    void run() {
        localExtractor = new LocalPatchesExtractor(this.patch_size, this.step_size);

        GroupedRandomSplitter<String, FImage> random = new GroupedRandomSplitter<>(trainingData, 80, 0, 20);

        assigner = makeAssigner(random.getTrainingDataset());

        featureExtractor = new PatchFeatureExtractor(assigner);

        ann = new LiblinearAnnotator<FImage, String>(featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        System.err.println("TRAINING");
        ann.train(random.getTrainingDataset());//.getTrainingDataset());

        //TESTING WITH SMALL TEST SET
        GroupedDataset<String, ListDataset<FImage>, FImage> testing = random.getTestDataset();

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

    public HardAssigner<double[], double[], IntDoublePair> makeAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset) {
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

    class LocalPatchesExtractor implements LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> {
        private int PATCH_SIZE, STEP;

        /**
         * Construct a local feature extractor with a patch size and step size. If the step size
         * is less than the patch size, the patches will overlap.
         *
         * @param patch size in pixels of patch, patches are square
         * @param step  length in pixels between the start of the patches
         */
        LocalPatchesExtractor(int patch, int step) {
            this.PATCH_SIZE = patch;
            this.STEP = step;
        }

        /**
         * Extracts a list of local features from an image, based on the patch size and patch step
         *
         * @param image extract local features from this
         * @return list of extracted local features
         */
        @Override
        public List<LocalFeature<SpatialLocation, DoubleFV>> extractFeature(FImage image) {
            List<LocalFeature<SpatialLocation, DoubleFV>> allPatches = new ArrayList<>();

            // Use OpenImaj RectangleSampler to generate patches
            // Cannot use subImageIterator as we need patch locations
            for (Rectangle patch : new RectangleSampler(image, STEP, STEP, PATCH_SIZE, PATCH_SIZE)) {
                // Feature vector is defined as the values of the patch pixels
                // Patches are constant size therefore feature vectors have constant dimensionality
                DoubleFV featureVector = zeroMean(new DoubleFV(image.extractROI(patch).getDoublePixelVector())).normaliseFV();
                // Location of feature is the location of the patch
                SpatialLocation location = new SpatialLocation(patch.x, patch.y);
                LocalFeature<SpatialLocation, DoubleFV> localFeature = new LocalFeatureImpl<>(location, featureVector);

                allPatches.add(localFeature);
            }

            Collections.shuffle(allPatches);
            return allPatches.subList(0, 10);
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

        @Override
        public Class<LocalFeature<SpatialLocation, DoubleFV>> getFeatureClass() {
            // Not needed
            return null;
        }
    }

    class PatchFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
        HardAssigner<double[], double[], IntDoublePair> assigner;

        public PatchFeatureExtractor(HardAssigner<double[], double[], IntDoublePair> assigner) {
            this.assigner = assigner;
        }


        @Override
        public DoubleFV extractFeature(FImage image) {
            BagOfVisualWords<double[]> bag = new BagOfVisualWords<double[]>(assigner);

            //            BlockSpatialAggregator<double[], SparseIntFV> spatial = new BlockSpatialAggregator<double[], SparseIntFV>(bag, 2, 2);
            //
            //            return spatial.aggregate(localExtractor.extractFeature(image), image.getBounds()).asDoubleFV();

            return bag.aggregate(localExtractor.extractFeature(image)).asDoubleFV();
        }

    }
}
