package rs25npk1;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.net.URISyntaxException;
import java.net.URL;

public class Main {
    // Used to force the downloading images from URLs
    // If false, initialise_data will look for unzipped image folders first
    //  before using the URLS
    private final boolean URLS = false;

    /**
     * Initialise the training and testing data
     *
     * @param trainingData grouped store of training images
     * @param testingData  list of images to classify
     */
    void initialise_data(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData) {

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
            trainingData = new VFSGroupDataset<>(trainingURI, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
        // Remove training folder if present
        trainingData.remove("training");

        try {
            testingData = new VFSListDataset<>(testingURI, ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            e.printStackTrace();
        }
    }
}
