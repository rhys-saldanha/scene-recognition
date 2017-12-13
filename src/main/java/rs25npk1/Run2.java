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
        System.err.println("Making Local Extractor");
        localExtractor = new LocalPatchesExtractor();
        System.err.println("Local Extractor Made");
        
        System.err.println("Making Assigner");
        GroupedRandomSplitter<String, FImage> random = new GroupedRandomSplitter<String, FImage>(trainingData, 15, 0, 0);
        assigner = makeAssigner(random.getTrainingDataset());
        System.err.println("Assigner Made");
        
        System.err.println("Making Feature Extractor");
        featureExtractor = new PatchFeatureExtractor(assigner);
        System.err.println("Feature Extractor Made");

        System.err.println("Making Linear Annotator");
        ann = new LiblinearAnnotator<FImage, String>(featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        System.err.println("Linear Annotator Made");
        
        System.err.println("Training!");
        ann.train(trainingData);
        System.err.println("Trained!");

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
            System.out.println("Classified");
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

        LocalPatchesExtractor extractor = new LocalPatchesExtractor();
        for (FImage i : groupedDataset) {
            patches.addAll(extractor.extractFeature(i));
        }
        
        // (try ~500 clusters to start)
        DoubleKMeans dkm = DoubleKMeans.createKDTreeEnsemble(500);
        DoubleCentroidsResult clusters = dkm.cluster(patches.toArray(new double[][]{}));

        return clusters.defaultHardAssigner();
    }


    class LocalPatchesExtractor implements LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> {

        @Override
        public List<LocalFeature<SpatialLocation, DoubleFV>> extractFeature(FImage image) {
            List<LocalFeature<SpatialLocation, DoubleFV>> allPatches = new ArrayList<>();

            RectangleSampler patches = new RectangleSampler(image, PATCH_STEP, PATCH_STEP, PATCH_SIZE, PATCH_SIZE);

            for (Rectangle patch : patches) {
                FImage area = image.extractROI(patch);

                //2D array to 1D array
                double[] vector = ArrayUtils.reshape(ArrayUtils.convertToDouble(area.pixels));
                DoubleFV featureVector = new DoubleFV(vector);
                //Location of rectangle is location of feature
                SpatialLocation location = new SpatialLocation(patch.x, patch.y);

                //Generate as a local feature for compatibility with other modules
                LocalFeature<SpatialLocation, DoubleFV> localFeature = new LocalFeatureImpl<SpatialLocation, DoubleFV>(location, featureVector);

                allPatches.add(localFeature);
            }
            return allPatches;
        }

        @Override
        public Class<LocalFeature<SpatialLocation, DoubleFV>> getFeatureClass() {
            // TODO Auto-generated method stub
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
