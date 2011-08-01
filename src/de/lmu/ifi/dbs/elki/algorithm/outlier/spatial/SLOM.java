package de.lmu.ifi.dbs.elki.algorithm.outlier.spatial;

import de.lmu.ifi.dbs.elki.algorithm.outlier.spatial.neighborhood.NeighborSetPredicate;
import de.lmu.ifi.dbs.elki.data.type.TypeInformation;
import de.lmu.ifi.dbs.elki.data.type.TypeUtil;
import de.lmu.ifi.dbs.elki.database.Database;
import de.lmu.ifi.dbs.elki.database.datastore.DataStoreFactory;
import de.lmu.ifi.dbs.elki.database.datastore.DataStoreUtil;
import de.lmu.ifi.dbs.elki.database.datastore.WritableDataStore;
import de.lmu.ifi.dbs.elki.database.ids.DBID;
import de.lmu.ifi.dbs.elki.database.ids.DBIDs;
import de.lmu.ifi.dbs.elki.database.query.distance.DistanceQuery;
import de.lmu.ifi.dbs.elki.database.relation.Relation;
import de.lmu.ifi.dbs.elki.distance.distancefunction.PrimitiveDistanceFunction;
import de.lmu.ifi.dbs.elki.distance.distancevalue.NumberDistance;
import de.lmu.ifi.dbs.elki.logging.Logging;
import de.lmu.ifi.dbs.elki.math.DoubleMinMax;
import de.lmu.ifi.dbs.elki.database.relation.MaterializedRelation;
import de.lmu.ifi.dbs.elki.result.outlier.BasicOutlierScoreMeta;
import de.lmu.ifi.dbs.elki.result.outlier.OutlierResult;
import de.lmu.ifi.dbs.elki.result.outlier.OutlierScoreMeta;
import de.lmu.ifi.dbs.elki.utilities.documentation.Description;
import de.lmu.ifi.dbs.elki.utilities.documentation.Reference;
import de.lmu.ifi.dbs.elki.utilities.documentation.Title;

/**
 * SLOM: a new measure for local spatial outliers
 * 
 * <p>
 * Reference:<br>
 * Sanjay Chawla and Pei Sun<br>
 * SLOM: a new measure for local spatial outliers<br>
 * in Knowledge and Information Systems 2005
 * </p>
 * 
 * This implementation works around some corner cases in SLOM, in particular
 * when an object has none or a single neighbor only (albeit the results will
 * still not be too useful then), which will result in divisions by zero.
 * 
 * @author Ahmed Hettab
 * 
 * @param <N> the type the spatial neighborhood is defined over
 * @param <O> the type of objects handled by the algorithm
 * @param <D> the type of Distance used for non spatial attributes
 */
@Title("SLOM: a new measure for local spatial outliers")
@Description("Spatial local outlier measure (SLOM), which captures the local behaviour of datum in their spatial neighbourhood")
@Reference(authors = "Sanjay Chawla and Pei Sun", title = "SLOM: a new measure for local spatial outliers", booktitle = "Knowledge and Information Systems 2005", url = "http://rp-www.cs.usyd.edu.au/~chawlarg/papers/KAIS_online.pdf")
public class SLOM<N, O, D extends NumberDistance<D, ?>> extends AbstractDistanceBasedSpatialOutlier<N, O, D> {
  /**
   * The logger for this class.
   */
  private static final Logging logger = Logging.getLogger(SLOM.class);

  /**
   * Constructor.
   * 
   * @param npred Neighborhood predicate
   * @param nonSpatialDistanceFunction Distance function to use on the
   *        non-spatial attributes
   */
  public SLOM(NeighborSetPredicate.Factory<N> npred, PrimitiveDistanceFunction<O, D> nonSpatialDistanceFunction) {
    super(npred, nonSpatialDistanceFunction);
  }

  /**
   * @param database Database to process
   * @param spatial Spatial Relation to use.
   * @param relation Relation to use.
   * @return Outlier detection result
   */
  public OutlierResult run(Database database, Relation<N> spatial, Relation<O> relation) {
    final NeighborSetPredicate npred = getNeighborSetPredicateFactory().instantiate(spatial);
    DistanceQuery<O, D> distFunc = getNonSpatialDistanceFunction().instantiate(relation);

    WritableDataStore<Double> modifiedDistance = DataStoreUtil.makeStorage(relation.getDBIDs(), DataStoreFactory.HINT_HOT | DataStoreFactory.HINT_TEMP, Double.class);
    // calculate D-Tilde
    for(DBID id : relation.iterDBIDs()) {
      double sum = 0;
      double maxDist = 0;
      int cnt = 0;

      final DBIDs neighbors = npred.getNeighborDBIDs(id);
      for(DBID neighbor : neighbors) {
        if(id.equals(neighbor)) {
          continue;
        }
        double dist = distFunc.distance(id, neighbor).doubleValue();
        sum += dist;
        cnt++;
        maxDist = Math.max(maxDist, dist);
      }
      if(cnt > 1) {
        modifiedDistance.put(id, ((sum - maxDist) / (cnt - 1)));
      }
      else {
        // Use regular distance when the d-tilde trick is undefined.
        // Note: this can be 0 when there were no neighbors.
        modifiedDistance.put(id, maxDist);
      }
    }

    // Second step - compute actual SLOM values
    DoubleMinMax slomminmax = new DoubleMinMax();
    WritableDataStore<Double> sloms = DataStoreUtil.makeStorage(relation.getDBIDs(), DataStoreFactory.HINT_STATIC, Double.class);

    for(DBID id : relation.iterDBIDs()) {
      double sum = 0;
      int cnt = 0;

      final DBIDs neighbors = npred.getNeighborDBIDs(id);
      for(DBID neighbor : neighbors) {
        if(neighbor.equals(id)) {
          continue;
        }
        sum += modifiedDistance.get(neighbor);
        cnt++;
      }
      double slom;
      if(cnt > 0) {
        // With and without the object itself:
        double avgPlus = (sum + modifiedDistance.get(id)) / (cnt + 1);
        double avg = sum / cnt;

        double beta = 0;
        for(DBID neighbor : neighbors) {
          final double dist = modifiedDistance.get(neighbor).doubleValue();
          if(dist > avgPlus) {
            beta += 1;
          }
          else if(dist < avgPlus) {
            beta -= 1;
          }
        }
        // Include object itself
        if(!neighbors.contains(id)) {
          final double dist = modifiedDistance.get(id).doubleValue();
          if(dist > avgPlus) {
            beta += 1;
          }
          else if(dist < avgPlus) {
            beta -= 1;
          }
        }
        beta = Math.abs(beta);
        // note: cnt == size of N(x), not N+(x)
        if(cnt > 1) {
          beta = Math.max(beta, 1.0) / (cnt - 1);
        }
        else {
          // Workaround insufficiency in SLOM paper - div by zero
          beta = 1.0;
        }
        beta = beta / (1 + avg);

        slom = beta * modifiedDistance.get(id);
      }
      else {
        // No neighbors to compare to - no score.
        slom = 0.0;
      }
      sloms.put(id, slom);
      slomminmax.put(slom);
    }

    Relation<Double> scoreResult = new MaterializedRelation<Double>("SLOM", "slom-outlier", TypeUtil.DOUBLE, sloms, relation.getDBIDs());
    OutlierScoreMeta scoreMeta = new BasicOutlierScoreMeta(slomminmax.getMin(), slomminmax.getMax(), 0.0, Double.POSITIVE_INFINITY);
    return new OutlierResult(scoreMeta, scoreResult);
  }

  @Override
  protected Logging getLogger() {
    return logger;
  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(getNeighborSetPredicateFactory().getInputTypeRestriction(), TypeUtil.NUMBER_VECTOR_FIELD);
  }

  /**
   * Parameterization class.
   * 
   * @author Ahmed Hettab
   * 
   * @apiviz.exclude
   * 
   * @param <N> Neighborhood type
   * @param <O> Data Object type
   * @param <D> Distance type
   */
  public static class Parameterizer<N, O, D extends NumberDistance<D, ?>> extends AbstractDistanceBasedSpatialOutlier.Parameterizer<N, O, D> {
    @Override
    protected SLOM<N, O, D> makeInstance() {
      return new SLOM<N, O, D>(npredf, distanceFunction);
    }
  }
}