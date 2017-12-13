package rs25npk1;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.net.URISyntaxException;
import java.net.URL;

class Main {
    static final boolean URLS = false;
    static VFSGroupDataset<FImage> trainingData;
    static VFSListDataset<FImage> testingData;

    Main() {
        initialise_data();
    }

    Main(VFSGroupDataset<FImage> training, VFSListDataset<FImage> testing) {
        trainingData = training;
        testingData = testing;
    }

    protected void initialise_data() {
        URL training = ClassLoader.getSystemResource("training");
        URL testing = ClassLoader.getSystemResource("testing");

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
            trainingData = new VFSGroupDataset<FImage>(trainingURI, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
        // Remove training folder
        trainingData.remove("training");

        try {
            testingData = new VFSListDataset<>(testingURI, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
    }
}
