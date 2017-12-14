package rs25npk1;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import javax.swing.plaf.basic.BasicInternalFrameTitlePane.SystemMenuBar;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import de.bwaldvogel.liblinear.SolverType;


public class Run2 extends Main {
    LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> localExtractor;
    HardAssigner<double[], double[], IntDoublePair> assigner;
    FeatureExtractor<DoubleFV, FImage> featureExtractor;
    LiblinearAnnotator<FImage, String> ann;

    public static void main(String[] args) {
        (new Run2()).run();
    }

    @Override
    void run() {
        localExtractor = new LocalPatchesExtractor(8, 4);

        GroupedRandomSplitter<String, FImage> random = new GroupedRandomSplitter<String, FImage>(trainingData, 80, 0, 20);

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

                String guesses = "";
                for (String s : guess) {
                    guesses = guesses + s;
                }

                if (guesses.equals(cls)) correct++;
                results.put((int) total, guesses);

                total++;
            }
        }
        
        System.out.println("Correct: " + correct + " | Total: " + total + " | Accuracy: " + (correct/total)*100);

        //        for (int i = 0; i < testing.size(); i++) {
        //            FImage image = testing.get(i);
        //            String[] guess = ann.classify(image).getPredictedClasses().toArray(new String[]{});
        //            
        //            String guesses = "";
        //            for (String s : guess) {
        //                guesses = guesses + s;
        //            }
        //            
        //            results.put(i, guesses);
        //        }

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
                DoubleFV featureVector = new DoubleFV(image.extractROI(patch).getDoublePixelVector());
                // Location of feature is the location of the patch
                SpatialLocation location = new SpatialLocation(patch.x, patch.y);
                LocalFeature<SpatialLocation, DoubleFV> localFeature = new LocalFeatureImpl<>(location, featureVector);

                allPatches.add(localFeature);
            }

            Collections.shuffle(allPatches);
            return allPatches.subList(0, 10);
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

            return bag.aggregate(localExtractor.extractFeature(image)).asDoubleFV().normaliseFV();
        }

    }
}
