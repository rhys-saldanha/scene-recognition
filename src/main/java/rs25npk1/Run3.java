package rs25npk1;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.ml.kernel.HomogeneousKernelMap.KernelType;
import org.openimaj.ml.kernel.HomogeneousKernelMap.WindowType;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;

public class Run3 implements OurClassifier {
    LiblinearAnnotator<FImage, String> ann;
    
    public void train(VFSGroupDataset<FImage> trainingData) {
     // Construct Dense SIFT extractor
        System.err.println("Constructing SIFT extractor...");
        DenseSIFT dsift = new DenseSIFT(5, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);        
        System.err.println("SIFT extractor constructed");

        System.err.println("Training quantiser...");
        HardAssigner<byte[], float[], IntFloatPair> assigner = 
                trainQuantiser(GroupedUniformRandomisedSampler.sample(trainingData, 30), pdsift);
        System.err.println("Quantiser trained");

        System.err.println("Constructing PHOW extractor...");
        FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);
        System.err.println("PHOW extractor made");
        
        System.err.println("Constructing KernelMap...");
        HomogeneousKernelMap kMap = new HomogeneousKernelMap(KernelType.Chi2, WindowType.Rectangular);
        System.err.println("KernelMap constructed");
        
        System.err.println("Constructing Feature extractor...");
        FeatureExtractor<DoubleFV, FImage> extractor2 = kMap.createWrappedExtractor(extractor);
        System.err.println("Feature extractor constructed");

        System.err.println("Constructing linear annotator...");
        ann = new LiblinearAnnotator<FImage, String>(
                extractor2, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        System.err.println("Linear annotator constructed");
    }

    public String classify(FImage f) {
        String[] guess = ann.classify(f).getPredictedClasses().toArray(new String[]{});

        String guesses = "";
        for (String s : guess) {
            guesses = guesses + s;
        }
        
        return guesses;
    }

//    void run() {
//
//        
//        System.err.println("Training...");
//        ann.train(training);
//        System.err.println("Trained!");
//
//
//        //        for (int i = 0; i < 20; i++) {
//        //            FImage im = testingData.get(i);
//        //            String fname = testingData.getID(i);
//        //            
//        //            String[] guess = ann.classify(im).getPredictedClasses().toArray(new String[]{});
//        //            String guesses = "";
//        //            for (String s : guess) {
//        //                guesses = guesses + s;
//        //            }
//        //            System.out.println(fname + " " + guesses);
//        //
//        //        }
//
//        Map<Integer, String> results = new HashMap<>();
//
//        System.err.println("Testing...");
//        double correct = 0, total = 0;
//        for (String cls : testing.getGroups()) {
//            //Loop through each face in the testing set
//            for (FImage im : testing.get(cls)) {
//                String[] guess = ann.classify(im).getPredictedClasses().toArray(new String[]{});
//
//                String guesses = "";
//                for (String s : guess) {
//                    guesses = guesses + s;
//                }
//
//                if (guesses.equals(cls)) correct++;
//                results.put((int) total, guesses);
//
//                total++;
//            }
//        }
//        System.err.println("Done!");
//        System.out.println("Accuracy: " + correct/total + "%");
//    }

    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(
            GroupedDataset<String, ListDataset<FImage>, FImage> sample, PyramidDenseSIFT<FImage> pdsift)
    {
        System.err.println("\tMaking list...");
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<LocalFeatureList<ByteDSIFTKeypoint>>();
        System.err.println("\tList made");

        System.err.println("\tAdding features to list...");
        for (Entry<String, ListDataset<FImage>> entry : sample.entrySet()) 
        {
            for(FImage image : entry.getValue()) 
            {
                pdsift.analyseImage(image.normalise());
                allkeys.add(pdsift.getByteKeypoints(0.005f));
            }
        }
        System.err.println("\tAll features added");


        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        System.err.println("\tCreate KDTreeEnsemble...");
        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        System.err.println("\tCreated");
        System.err.println("\tConstructing datasource...");
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<ByteDSIFTKeypoint, byte[]>(allkeys);
        System.err.println("\tDatasource made");
        System.err.println("\tClustering...");
        ByteCentroidsResult result = km.cluster(datasource);
        System.err.println("\tClustered!");

        return result.defaultHardAssigner();
    }

    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<byte[]>(assigner);

            BlockSpatialAggregator<byte[], SparseIntFV> spatial = new BlockSpatialAggregator<byte[], SparseIntFV>(
                    bovw, 2, 2);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }

}
