package rs25npk1;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class Main {
    // Used to force the downloading images from URLs
    // If false, initialise_data will look for unzipped image folders first
    //  before using the URLS
    private static final boolean URLS = false;

    private static VFSGroupDataset<FImage> initialise_training() {
        VFSGroupDataset<FImage> trainingData = null;
        URL training = ClassLoader.getSystemResource("training");
        String trainingURI = null;
        if (training == null || URLS) {
            System.err.println("Using training URL");
            trainingURI = "zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip";
        } else {
            try {
                trainingURI = training.toURI().getPath();
            } catch (URISyntaxException e) {
                e.printStackTrace();
            }
        }
        try {
            trainingData = new VFSGroupDataset<>(trainingURI, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
        // Remove training folder if present
        assert trainingData != null;
        trainingData.remove("training");
        return trainingData;
    }

    private static VFSListDataset<FImage> initialise_testing() {
        VFSListDataset<FImage> testingData = null;
        URL testing = ClassLoader.getSystemResource("testing");
        String testingURI = null;
        if (testing == null || URLS) {
            System.err.println("Using testing URL");
            testingURI = "zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip";
        } else {
            try {
                testingURI = testing.toURI().getPath();
            } catch (URISyntaxException e) {
                e.printStackTrace();
            }
        }
        try {
            testingData = new VFSListDataset<>(testingURI, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
        return testingData;
    }

    public static void main(String[] args) {
        VFSGroupDataset<FImage> training = initialise_training();
        VFSListDataset<FImage> testing = initialise_testing();


//        GroupedRandomSplitter<String, FImage> random = new GroupedRandomSplitter<>(training, 8, 0, 2);

        classify_to_file(training, testing, new KNN(), "run1");
        classify_to_file(training, testing, new LIN(), "run2");
        classify_to_file(training, testing, new Run3(), "run3");
    }

    private static void classify_to_file(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing, Classifier classifier, String name) {
        // TRAIN
        System.err.println("Training");
        classifier.train(training);
        System.err.println("Trained");

        //TESTING WITH SMALL TEST SET
        System.err.println("Testing");

        try (Writer writer = new FileWriter(new File(name + ".txt"))) {
            IntStream.range(0, testing.size()).parallel().forEach(i -> {
                FImage f = testing.getInstance(i);
                try {
                    writer.write(String.format("%s %s\n", testing.getID(i).split("/")[1], classifier.classify(f)));
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.err.println("FIN");
    }
}
