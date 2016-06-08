package de.lmu.ifi.dbs.elki.algorithm.statistics;

import de.lmu.ifi.dbs.elki.data.*;
import de.lmu.ifi.dbs.elki.database.query.knn.KNNQuery;
import de.lmu.ifi.dbs.elki.distance.distancefunction.NumberVectorDistanceFunction;
import de.lmu.ifi.dbs.elki.math.MathUtil;
import de.lmu.ifi.dbs.elki.utilities.random.RandomFactory;

import java.util.*;

/**
 * extends the Hopkins Statistics class to use binary vectors as uniform data (1,0,0,1...)
 * useful for evaluating the clustering tendency with a set distance function such as Jaccard's index
 */
public class HopkinsStatisticClusteringTendencyBinaryData extends HopkinsStatisticClusteringTendency {

    /**
     * Constructor.
     *
     * @param distanceFunction Distance function
     * @param samplesize       Sample size
     * @param random           Random generator
     * @param rep              Number of repetitions
     * @param k                Nearest neighbor to use
     * @param minima           Data space minima, may be {@code null} (get from data).
     * @param maxima           Data space minima, may be {@code null} (get from data).
     */
    public HopkinsStatisticClusteringTendencyBinaryData(NumberVectorDistanceFunction<? super NumberVector> distanceFunction, int samplesize, RandomFactory random, int rep, int k, double[] minima, double[] maxima) {
        super(distanceFunction, samplesize, random, rep, k, minima, maxima);
    }

    @Override
    protected double computeNNForUniformData(KNNQuery<NumberVector> knnQuery, double[] min, double[] extend) {
        final Random rand = random.getSingleThreadedRandom();
        final int dim = min.length;

        double u = 0.;
        for(int i = 0; i < sampleSize; i++) {
            // New random vector
            List<Integer> setDimensions = new ArrayList<>();
            for(int d = 0; d < dim; d++)
                if (rand.nextBoolean())
                    setDimensions.add(d);

            int[] indices = new int[setDimensions.size()];
            short[] values = new short[indices.length];
            Iterator<Integer> iterator = setDimensions.iterator();
            for (int j = 0; iterator.hasNext(); j++) {
                indices[j] = iterator.next();
                values[j] = 1;
            }
            SparseNumberVector vector = new SparseShortVector(indices, values, dim);
            double kdist = knnQuery.getKNNForObject(vector, k).getKNNDistance();
            u += MathUtil.powi(kdist, dim);
        }
        return u;
    }


    public static class Parametizer extends HopkinsStatisticClusteringTendency.Parameterizer {

        @Override
        protected HopkinsStatisticClusteringTendency makeInstance() {
            return new HopkinsStatisticClusteringTendencyBinaryData(distanceFunction, sampleSize, random, rep, k, minima, maxima);
        }
    }
}

