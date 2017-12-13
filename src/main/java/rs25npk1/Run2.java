package rs25npk1;

import java.util.ArrayList;
import java.util.List;

import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;

public class Run2 extends Main {

    @Override
    void run() {
        // TODO Auto-generated method stub

    }

    /*
     *
     * Step 1 : Create HardAssigner  (DING DING DING)
     * Step 2 : Give HardAssigner to extractor to get patches
     * Step 3 : Extractor to LiblinearAnnotator
     * Step 4 : Train + Test
     *
     */

    private int PATCH_SIZE = 8;
    private int PATCH_STEP = 4;

    public HardAssigner<double[], double[], IntDoublePair> makeAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> data) {
        List<LocalFeature<SpatialLocation, DoubleFV>> patches = new ArrayList<>();
        
        PatchesExtractor extractor = new PatchesExtractor();
        for (FImage i : data) {
            patches.addAll(extractor.extractFeature(i));
        }
        
        DoubleKMeans dkm = DoubleKMeans.createKDTreeEnsemble(300);
        DoubleCentroidsResult clusters = dkm.cluster(patches.toArray(new double[][]{}));
        
        return clusters.defaultHardAssigner();
    }
    
    
    class PatchesExtractor implements LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> {

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
}