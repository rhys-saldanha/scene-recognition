package rs25npk1;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.net.URISyntaxException;
import java.net.URL;

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
            System.out.println("Using training URL");
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
        trainingData.remove("training");
        return trainingData;
    }

    private static VFSListDataset<FImage> initialise_testing() {
        VFSListDataset<FImage> testingData = null;
        URL testing = ClassLoader.getSystemResource("testing");
        String testingURI = null;
        if (testing == null || URLS) {
            System.out.println("Using testing URL");
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
//        VFSListDataset<FImage> testing = initialise_testing();


        GroupedRandomSplitter<String, FImage> random = new GroupedRandomSplitter<>(training, 80, 0, 20);

        Classifier classifier = new LIN();
//        Classifier classifier = new KNN();
//        Classifier classifier = new Run3();

        // TRAIN LIBLINEARANNOTATOR
        System.err.println("Training");
        classifier.train(random.getTrainingDataset());
        System.err.println("Trained");

        //TESTING WITH SMALL TEST SET

        System.err.println("Testing");
        double correct = 0, total = 0;
        for (String cls : random.getTestDataset().getGroups()) {
            //Loop through each face in the testing set
            for (FImage im : random.getTestDataset().get(cls)) {
                String guess = classifier.classify(im);
                if (guess.equals(cls)) correct++;
                total++;
            }
        }

        System.out.println("Correct: " + correct + " | Total: " + total + " | Accuracy: " + (correct / total) * 100);

//        System.err.println("Writing");
//        try (Writer writer = new FileWriter(new File("run2.txt"))) {
//            System.err.println("Classified");
//            for (Map.Entry<Integer, String> entry : results.entrySet()) {
//                writer.write(String.format("%s %s\n", entry.getKey(), results.get(entry.getKey())));
//            }
//        } catch (IOException e) {
//            e.printStackTrace();
//        }

        System.err.println("Fin!");
    }
}
