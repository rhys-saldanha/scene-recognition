package rs25npk1;

import java.util.ArrayList;
import java.util.List;

import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureExtractor;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.util.array.ArrayUtils;

public class Run2 extends Main{

	@Override
	void run() {
		// TODO Auto-generated method stub
		
	}
	
	/*
	 * 
	 * Step 1 : Create HardAssigner 
	 * Step 2 : Give HardAssigner to extractor to get patches
	 * Step 3 : Extractor to LiblinearAnnotator
	 * Step 4 : Train + Test
	 * 
	 */
    
	private int PATCH_SIZE = 8;
	private int PATCH_STEP = 4;
	
	public class PatchesExtractor implements LocalFeatureExtractor<LocalFeature<SpatialLocation, DoubleFV>, FImage> {

		@Override
		public List<LocalFeature<SpatialLocation, DoubleFV>> extractFeature(FImage image) {
		    List<LocalFeature<SpatialLocation, DoubleFV>> allPatches = new ArrayList<>();
			
			RectangleSampler patches = new RectangleSampler(image, PATCH_STEP, PATCH_STEP, PATCH_SIZE, PATCH_SIZE);
			
	        for(Rectangle patch : patches){
	            FImage area = image.extractROI(patch);

	            //2D array to 1D array
	            double[] vector = ArrayUtils.reshape(ArrayUtils.convertToDouble(area.pixels));
	            DoubleFV featureV = new DoubleFV(vector);
	            //Location of rectangle is location of feature
	            SpatialLocation sl = new SpatialLocation(patch.x, patch.y);
	            
	            //Generate as a local feature for compatibility with other modules
	            LocalFeature<SpatialLocation, DoubleFV> lf = new LocalFeatureImpl<SpatialLocation, DoubleFV>(sl,featureV);

	            allPatches.add(lf);
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
