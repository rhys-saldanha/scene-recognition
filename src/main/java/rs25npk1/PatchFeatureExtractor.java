package rs25npk1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntDoublePair;

class PatchFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {

    private HardAssigner<double[], double[], IntDoublePair> assigner;
    private LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> localExtractor;

    PatchFeatureExtractor(HardAssigner<double[], double[], IntDoublePair> assigner, LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> localExtractor) {
        this.assigner = assigner;
        this.localExtractor = localExtractor;
    }

    @Override
    public DoubleFV extractFeature(FImage image) {
        BagOfVisualWords<double[]> bag = new BagOfVisualWords<>(assigner);
        BlockSpatialAggregator<double[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bag, 2, 2);
        return spatial.aggregate(localExtractor.extractFeature(image), image.getBounds()).asDoubleFV();
//        return bag.aggregate(localExtractor.extractFeature(image)).asDoubleFV();
    }

}
