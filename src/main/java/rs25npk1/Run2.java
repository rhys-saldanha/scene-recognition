package rs25npk1;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;

import de.bwaldvogel.liblinear.SolverType;


public class Run2 extends Main {
    LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> localExtractor;
    HardAssigner<double[], double[], IntDoublePair> assigner;
    FeatureExtractor<DoubleFV, FImage> featureExtractor;
    LiblinearAnnotator<FImage, String> ann;

    private int PATCH_SIZE = 8;
    private int PATCH_STEP = 4;

    public static void main(String[] args) {
        (new Run2()).run();
    }

    @Override
    void run() {
        localExtractor = new LocalPatchesExtractor();

        GroupedRandomSplitter<String, FImage> random = new GroupedRandomSplitter<String, FImage>(trainingData, 15, 0, 0);
        assigner = makeAssigner(random.getTrainingDataset());

        featureExtractor = new PatchFeatureExtractor(assigner);

        ann = new LiblinearAnnotator<FImage, String>(featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        ann.train(trainingData);

        //TESTING WITH SMALL TEST SET
        testingData = (VFSListDataset<FImage>) testingData.subList(0, 100);

        System.err.println("Testing");
        Map<String, String> results = new HashMap<>();
        for (int i = 0; i < testingData.size(); i++) {
            FImage image = testingData.get(i);
            String name = testingData.getID(i);
            String guess = ann.classify(image).toString();

            results.put(name, guess);
        }

        try (Writer writer = new FileWriter(new File("run2.txt"))) {
            System.err.println("Classified");
            IntStream.range(0, testingData.size()).forEach(index -> {
                try {
                    writer.write(String.format("%s %s\n", index, results.get(index)));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
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

        System.err.println(".cluster() is what's slow (more mem maybe?)");

        DoubleCentroidsResult clusters = dkm.cluster(keys.toArray(new double[][]{}));

        System.err.println(".cluster() done. Ayyyyyy");

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
            return allPatches;
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

            BlockSpatialAggregator<double[], SparseIntFV> spatial = new BlockSpatialAggregator<double[], SparseIntFV>(bag, 2, 2);

            return spatial.aggregate(localExtractor.extractFeature(image), image.getBounds()).normaliseFV();
        }

    }
}
