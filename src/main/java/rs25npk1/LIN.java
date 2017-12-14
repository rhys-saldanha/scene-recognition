package rs25npk1;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import java.util.ArrayList;
import java.util.List;


class LIN implements Classifier {
    private LiblinearAnnotator<FImage, String> ann;
    private int patch_size, step_size;

    LIN() {
        this(8, 4);
    }

    LIN(int patch_size, int step_size) {
        this.patch_size = patch_size;
        this.step_size = step_size;
    }

    @Override
    public void train(GroupedDataset<String, ListDataset<FImage>, FImage> trainingData) {
        LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> localExtractor = new LocalPatchesExtractor(this.patch_size, this.step_size);

        HardAssigner<double[], double[], IntDoublePair> assigner = makeAssigner(trainingData);

        FeatureExtractor<DoubleFV, FImage> featureExtractor = new PatchFeatureExtractor(assigner, localExtractor);

        ann = new LiblinearAnnotator<>(featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);

        System.err.println("Training");
        ann.train(trainingData);
        System.err.println("Training: complete");
    }

    private HardAssigner<double[], double[], IntDoublePair> makeAssigner(GroupedDataset<String, ListDataset<FImage>, FImage> groupedDataset) {
        List<LocalFeatureList<LocalFeature<SpatialLocation, DoubleFV>>> patches = new ArrayList<>();

        LocalPatchesExtractor extractor = new LocalPatchesExtractor(8, 4);
        for (FImage i : groupedDataset) {
            patches.add(extractor.extractFeature(i));
        }

//        List<double[]> keys = new ArrayList<>();
//        for (LocalFeature p : patches) {
//            keys.add(p.getFeatureVector().asDoubleVector());
//        }

        // (try ~500 clusters to start)

        DoubleKMeans dkm = DoubleKMeans.createKDTreeEnsemble(500);

        DataSource<double[]> datasource = new LocalFeatureListDataSource<>(patches);

        DoubleCentroidsResult clusters = dkm.cluster(datasource);

        return clusters.defaultHardAssigner();
    }

    @Override
    public String classify(FImage f) {
        return ann.classify(f).getPredictedClasses().toArray(new String[]{})[0];
    }
}
