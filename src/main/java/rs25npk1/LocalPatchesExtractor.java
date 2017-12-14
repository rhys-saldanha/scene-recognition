package rs25npk1;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

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
            DoubleFV featureVector = zeroMean(new DoubleFV(image.extractROI(patch).getDoublePixelVector())).normaliseFV();
            // Location of feature is the location of the patch
            SpatialLocation location = new SpatialLocation(patch.x, patch.y);
            LocalFeature<SpatialLocation, DoubleFV> localFeature = new LocalFeatureImpl<>(location, featureVector);

            allPatches.add(localFeature);
        }

        Collections.shuffle(allPatches);
        return allPatches.subList(0, 10);
    }

    private DoubleFV zeroMean(DoubleFV feature) {
        feature = feature.clone();
        double mean = 0;
        for (double d : feature.values) {
            mean += d;
        }
        mean /= feature.values.length;
        double[] newValues = new double[feature.values.length];
        for (int i = 0; i < feature.values.length; i++) {
            newValues[i] = feature.values[i] - mean;
        }
        return new DoubleFV(newValues);
    }

    @Override
    public Class<LocalFeature<SpatialLocation, DoubleFV>> getFeatureClass() {
        // Not needed
        return null;
    }
}